#!/usr/bin/env python3
"""Freeze an input-selected train/validation feasibility pilot; test stays locked."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from rdkit import Chem, RDLogger
from prepare_rerank_pilot import joined_groups
from rerank_analysis import PARSER

RDLogger.DisableLog('rdApp.*')
SEED = 'flower-targeted-feasibility-20261029-v1'
SMARTS = {
    'ester': '[CX3](=[OX1])[OX2][#6]',
    'carboxylic_acid': '[CX3](=[OX1])[OX2H1]',
}
QUERIES = {k:Chem.MolFromSmarts(v) for k,v in SMARTS.items()}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def motifs(smiles):
    mol=Chem.MolFromSmiles(smiles,PARSER)
    if mol is None:
        return {'parseable':False, 'motifs':[]}
    # Heavy-atom view only for atom H counts/functional group detection.
    mol=Chem.RemoveHs(mol)
    atoms=list(mol.GetAtoms())
    protonated_donor=any(a.GetAtomicNum() in (7,8) and a.GetFormalCharge()>0 and a.GetTotalNumHs()>0 for a in atoms)
    hetero_acceptors=sum(a.GetAtomicNum() in (7,8) and a.GetFormalCharge()<=0 for a in atoms)
    hydroxide=any(a.GetAtomicNum()==8 and a.GetFormalCharge()==-1 and a.GetDegree()==0 and a.GetTotalNumHs()==1 for a in atoms)
    anionic_hetero=any(a.GetAtomicNum() in (7,8) and a.GetFormalCharge()<0 for a in atoms)
    hydrogen_halide=any(a.GetAtomicNum() in (9,17,35,53) and a.GetFormalCharge()==0 and a.GetTotalNumHs()>0 for a in atoms)
    acid=hydrogen_halide or mol.HasSubstructMatch(QUERIES['carboxylic_acid'])
    flags=[]
    if protonated_donor and hetero_acceptors>=2:flags.append('protonated_donor_competing_acceptors')
    if acid and anionic_hetero:flags.append('acid_anionic_heteroatom')
    if hydroxide and mol.HasSubstructMatch(QUERIES['ester']):flags.append('hydroxide_ester')
    return {'parseable':True,'motifs':flags,'atoms':Chem.MolFromSmiles(smiles,PARSER).GetNumAtoms()}


def ordered(rows,role):
    return sorted(rows,key=lambda r:digest(f'{SEED}:{role}:{r["source_split"]}:{r["source_row_id"]}'))


def enriched(rows,role,n):
    targeted=[r for r in rows if r['input_motifs']]
    chosen=[dict(r,selection_cohort='motif') for r in ordered(targeted,role+':motif')[:n//2]]
    if len(chosen)!=n//2:raise ValueError(f'Insufficient motif rows for {role}')
    ids={r['source_row_id'] for r in chosen}
    chosen += [dict(r,selection_cohort='random_remainder') for r in ordered([r for r in rows if r['source_row_id'] not in ids],role+':random')[:n-len(chosen)]]
    assert len(chosen)==n
    return sorted(chosen,key=lambda r:r['source_row_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/reranking/targeted-pilot-20261029'))
    p.add_argument('--index-root',type=Path,default=Path('results/reranking/pilot-v2'))
    p.add_argument('--data',type=Path,default=Path('data/flower_new_dataset'))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'selection.json').exists(),'Selection already frozen'
    metadata={}
    for part in ('train','val','test'):
        metadata[part]=[json.loads(s) for s in (args.index_root/f'{part}-input-index.jsonl').open()]
    joined_groups(sum(metadata.values(),[]))
    old=json.loads((args.index_root/'selection.json').read_text())
    held=json.loads((args.index_root/'holdout-rows.json').read_text())
    held_ids={r['source_row_id'] for r in held}
    protected={r['split_group'] for r in metadata['test'] if r['source_row_id']<20152 or r['source_row_id'] in held_ids}
    val_groups={r['split_group'] for r in metadata['val']}
    eligible_train=[r for r in metadata['train'] if r['split_group'] not in protected|val_groups]
    eligible_val=[r for r in metadata['val'] if r['split_group'] not in protected]
    eligibility={'train':{r['source_row_id']:r for r in eligible_train},'val':{r['source_row_id']:r for r in eligible_val}}
    source_hashes={}
    parse_failures=[]
    for part,lookup in eligibility.items():
        h=hashlib.sha256()
        with (args.data/f'{part}.txt').open('rb') as stream:
            for index,raw in enumerate(stream):
                h.update(raw)
                if index not in lookup:continue
                row=lookup[index];line=raw.decode()
                assert hashlib.sha256(raw).hexdigest()==row['source_line_sha256']
                input_smiles=line.split('|',1)[0].split('>>',1)[0]
                result=motifs(input_smiles)
                row['input_motifs']=result['motifs'];row['input_parseable']=result['parseable']
                row['input_atoms']=result.get('atoms')
                if not result['parseable']:parse_failures.append([part,index])
        source_hashes[part]=h.hexdigest()
        assert h.hexdigest()==old['source_sha256'][part]
        print(part,'inputs mined',len(lookup),flush=True)
    assert not parse_failures,'Input parsing failure: keep explicit; resolve before freezing'
    # Group assignment precedes within-role sampling. All members share role.
    audit_pool=[r for r in eligible_val if int(digest(SEED+':val-role:'+r['split_group'])[:8],16)%2==0]
    tune_pool=[r for r in eligible_val if int(digest(SEED+':val-role:'+r['split_group'])[:8],16)%2==1]
    parts={
        'fit':enriched(eligible_train,'fit',1024),
        'tune':enriched(tune_pool,'tune',512),
        'audit':sorted([dict(r,selection_cohort='random_audit') for r in ordered(audit_pool,'audit')[:1024]],key=lambda r:r['source_row_id']),
    }
    for name,rows in parts.items():
        assert len(rows)=={'fit':1024,'tune':512,'audit':1024}[name]
        for i,r in enumerate(rows):r['row_id']=i
    for a,b in [('fit','tune'),('fit','audit'),('tune','audit')]:
        for key in ('split_group','reactant_key','sequence_id'):
            assert not ({r[key] for r in parts[a]} & {r[key] for r in parts[b]})
    assert all(r['split_group'] not in protected for rows in parts.values() for r in rows)
    (args.out/'inputs').mkdir(exist_ok=True)
    for part in ('train','val'):
        names=['fit'] if part=='train' else ['tune','audit']
        maps={name:{r['source_row_id']:r for r in parts[name]} for name in names}
        outputs={name:(args.out/'inputs'/f'{name}.txt').open('w') for name in names}
        try:
            with (args.data/f'{part}.txt').open() as stream:
                for index,line in enumerate(stream):
                    for name in names:
                        if index in maps[name]:
                            assert digest(line)==maps[name][index]['source_line_sha256']
                            outputs[name].write(line)
        finally:
            for f in outputs.values():f.close()
    for name,rows in parts.items():
        (args.out/f'{name}-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    receipt={
        'status':'frozen_before_candidate_generation','seed':SEED,'sample_size':64,'primary_k':10,
        'stage':'feasibility, not powered confirmation or full benchmark estimate',
        'source_sha256':source_hashes,
        'input_index_sha256':{part:sha(args.index_root/f'{part}-input-index.jsonl') for part in metadata},
        'reserved_manifest_sha256':sha(args.index_root/'holdout-rows.json'),
        'selection_script_sha256':sha(__file__),
        'source_pool_sizes':{part:len(rows) for part,rows in metadata.items()},
        'eligible_pool_sizes':{'fit':len(eligible_train),'tune':len(tune_pool),'audit':len(audit_pool)},
        'motif_definitions':{'smarts':SMARTS,'logic':'Positive N/O with H and >=2 neutral/negative N/O atoms; acid (HX/carboxylic) plus negative N/O; hydroxide plus ester. Input-only, no reaction outcome or predictions.'},
        'grouping':'Recomputed transitive same normalized input OR source-local network ID over existing train-pool/all-val/all-test input metadata',
        'exclusions':'Any component touching inspected test prefix or reserved512; fit excludes all validation components',
        'val_role_assignment':'hash(seed:val-role:component) modulo2 before row sampling; 0audit,1tune',
        'development_sampling':'Half motif-selected, half hash-random from remaining rows; cohorts stored; not representative rates',
        'audit_sampling':'Hash-random eligible validation rows within independently assigned audit components; no correctness/rank/motif enrichment',
        'reserved_test_targets_accessed':0,'unprocessed_test_targets_accessed':0,
        'gpu_budget_seconds':7200,'primary_method':'learned_targeted',
        'parts':{},
    }
    for name,rows in parts.items():
        receipt['parts'][name]={
            'source_split':rows[0]['source_split'],'rows':len(rows),
            'groups':len({r['split_group'] for r in rows}),
            'inputs_sha256':sha(args.out/'inputs'/f'{name}.txt'),
            'row_manifest_sha256':sha(args.out/f'{name}-rows.json'),
            'selection_cohorts':dict(Counter(r['selection_cohort'] for r in rows)),
            'motifs_overlap':dict(Counter(m for r in rows for m in r['input_motifs'])),
            'mean_input_atoms':sum(r['input_atoms'] for r in rows)/len(rows),
            'max_input_atoms':max(r['input_atoms'] for r in rows),
        }
    (args.out/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (args.out/'selection.json').chmod(0o444)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
