#!/usr/bin/env python3
"""Freeze a diverse 50-case diagnostic sample and prepare review evidence."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from audit_recorded_targets import mapped_change, normalized
from rerank_analysis import PARSER
RDLogger.DisableLog('rdApp.*')

SEED = 'flower-unmatched-review-20261029-v1'
QUOTAS = {'different_heavy_connectivity':24, 'same_heavy_bonds_H_charge_difference':16,
          'same_connectivity_bond_order_or_H_charge_difference':9, 'no_saved_candidate':1}


def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell(row):
    c = row['candidates']
    gap = c[0]['count']-c[1]['count'] if len(c)>1 else None
    band = ('none' if not c else 'near_tie' if gap is not None and gap<=4 else
            'dominant' if row['top1_count']>=48 else 'middle')
    return (row['comparison']['category'], 'in_top10' if row['strict_top10'] else 'outside_top10',
            band, 'no_change' if row['top1_no_change'] else 'changed')


def select(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[cell(row)].append(row)
    quotas = {}
    for category, total in QUOTAS.items():
        keys = sorted(k for k in cells if k[0]==category)
        assert len(keys)<=total
        allocation = {key:1 for key in keys}
        remaining = total-len(keys)
        weights = {key:len(cells[key])-1 for key in keys}
        denom = sum(weights.values())
        ideal = {key:remaining*weights[key]/denom if denom else 0 for key in keys}
        for key in keys:
            allocation[key]+=int(ideal[key])
        extra = total-sum(allocation.values())
        for key in sorted(keys,key=lambda k:(-(ideal[k]-int(ideal[k])),hash_value(SEED+repr(k))))[:extra]:
            allocation[key]+=1
        quotas.update(allocation)
    selected, used_networks = [], set()
    for key in sorted(cells,key=lambda k:(len(cells[k])/quotas[k],hash_value(SEED+repr(k)))):
        candidates = sorted(cells[key],key=lambda r:hash_value(SEED+':'+str(r['row_id'])))
        chosen = []
        for r in candidates:
            if r['sequence_id'] not in used_networks:
                chosen.append(r);used_networks.add(r['sequence_id'])
                if len(chosen)==quotas[key]:break
        if len(chosen)<quotas[key]:
            chosen_ids={r['row_id'] for r in chosen}
            for r in candidates:
                if r['row_id'] not in chosen_ids:
                    chosen.append(r);chosen_ids.add(r['row_id']);used_networks.add(r['sequence_id'])
                    if len(chosen)==quotas[key]:break
        assert len(chosen)==quotas[key]
        selected.extend(chosen)
    assert len(selected)==50 and len({r['row_id'] for r in selected})==50
    assert Counter(r['comparison']['category'] for r in selected)==QUOTAS
    return sorted(selected,key=lambda r:r['row_id']), cells, quotas


def fragments(s):
    if s is None:return Counter()
    mol=Chem.MolFromSmiles(s,PARSER)
    return Counter(normalized(f) for f in Chem.GetMolFrags(mol,asMols=True))


def drawing(s):
    if not s:return None
    mol=Chem.MolFromSmiles(s,PARSER)
    for a in mol.GetAtoms():a.SetAtomMapNum(0)
    return Chem.RemoveHs(mol)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unmatched',type=Path,default=Path('results/label-audit-20152/unmatched-top1.jsonl'))
    p.add_argument('--records',type=Path,default=Path('results/reranking/failure-study/records.jsonl'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Keep the frozen sample; use a fresh output path')
    rows=[json.loads(s) for s in args.unmatched.open()]
    assert len(rows)==435 and all(not r['observed_top1'] for r in rows)
    chosen,cells,quotas=select(rows)
    assert [r['row_id'] for r in chosen]==[r['row_id'] for r in select(list(reversed(rows)))[0]]
    known=[json.loads(s) for s in args.records.open()]
    assert len(known)==20152 and max(r['row_id'] for r in known)==20151
    by_id={r['row_id']:r for r in known}
    networks=defaultdict(list)
    for r in known:networks[r['sequence_id']].append(r)
    args.out.mkdir(parents=True)
    (args.out/'dossiers').mkdir();(args.out/'figures').mkdir();(args.out/'reviews').mkdir()
    manifest={'status':'frozen_before_detailed_review','seed':SEED,'population':435,'sample':50,
        'purpose':'mechanism diagnosis, not prevalence or performance estimation',
        'selection':'One per nonempty category/coverage/confidence/no-change cell; distribute remaining category quota by largest remainder proportional to remaining population. Fixed hash order within cells, rare cells first, prefer distinct network IDs.',
        'confidence_definition':'near_tie: first-minus-second count <=4; dominant: otherwise top1 count >=48/64; middle: remaining candidates; none: no candidate',
        'category_quotas':QUOTAS,'source_sha256':{'unmatched':file_hash(args.unmatched),'records':file_hash(args.records),'script':file_hash(__file__)},
        'test_lines':[r['test_line'] for r in chosen],
        'cell_allocation':[{'cell':list(k),'population':len(cells[k]),'selected':quotas[k]} for k in sorted(cells)],
        'coverage':{'networks':len({r['sequence_id'] for r in chosen}),'reference_in_top10':sum(r['strict_top10'] for r in chosen),
            'reference_outside_top10':sum(not r['strict_top10'] for r in chosen),'top1_no_change':sum(r['top1_no_change'] for r in chosen),
            'reference_no_change':sum(r['reference_no_change'] for r in chosen),'confidence':dict(Counter(cell(r)[2] for r in chosen))}}
    (args.out/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.out/'selection.json').chmod(0o444)
    dossier_index=[]
    for ordinal,r in enumerate(chosen,1):
        source=by_id[r['row_id']];network=networks[r['sequence_id']]
        graph=defaultdict(set)
        for n in network:graph[n['reactant']].add(n['target'])
        distance={r['input']:0};queue=deque([r['input']])
        while queue:
            node=queue.popleft()
            for dest in graph.get(node,[]):
                if dest not in distance:distance[dest]=distance[node]+1;queue.append(dest)
        inp,ref=source['reaction'].split('>>')
        fs=[fragments(r['input']),fragments(r['top1']),fragments(r['reference'])]
        common=fs[0]&fs[1]&fs[2] if r['top1'] else fs[0]&fs[2]
        visible=[f-common for f in fs]
        smiles=['.'.join(s for s,n in f.items() for _ in range(n)) for f in visible]
        stripped=[Chem.MolToSmiles(drawing(s)) if s else None for s in smiles]
        evidence=dict(r,review_id=f'R{ordinal:02d}',selection_cell=list(cell(r)),
            full_mapped_reaction=source['reaction'],sample_metrics=source['metrics'],sample_count=source['sample_count'],
            saved_distinct_candidates=source['distinct_saved_candidates'],reference_count=source['metrics'][0],
            reference_edits=mapped_change(inp,ref),reference_in_any_saved_sample=source['target_rank'] is not None,
            top1_min_recorded_steps_from_input=distance.get(r['top1']),
            input_predecessors=[{'test_line':n['row_id']+1,'input':n['reactant']} for n in network if n['target']==r['input'] and n['reactant']!=r['input']],
            reference_successors=[{'test_line':n['row_id']+1,'target':n['target']} for n in network if n['reactant']==r['reference']],
            same_input_references=[{'test_line':n['row_id']+1,'target':n['target']} for n in network if n['reactant']==r['input']],
            display={'omitted_common_fragments':dict(common),'input_changed_fragments':stripped[0],
                     'top1_changed_fragments':stripped[1],'reference_changed_fragments':stripped[2]},
            figure=f'figures/case-{r["test_line"]:05d}.png',
            scope='Only already-inspected prefix labels; no external experimental provenance or kinetic/energy measurement.')
        path=args.out/'dossiers'/f'case-{r["test_line"]:05d}.json'
        path.write_text(json.dumps(evidence,indent=2)+'\n')
        options=Draw.MolDrawOptions();options.legendFontSize=18;options.bondLineWidth=2
        mols=[drawing(s) for s in smiles]
        legends=[f'{evidence["review_id"]} | line {r["test_line"]} | supplied input',
                 f'Top 1 | {r["top1_count"]}/64' if r['top1'] else 'No saved candidate',
                 f'Reference | rank {r["target_rank"] or "absent"} | {source["metrics"][0]}/64']
        img=Draw.MolsToGridImage(mols,molsPerRow=3,subImgSize=(650,650),legends=legends,drawOptions=options)
        img.save(str(args.out/evidence['figure']))
        full=Draw.MolsToGridImage([drawing(r['input'])],molsPerRow=1,subImgSize=(1500,850),
                                 legends=[f'{evidence["review_id"]} | complete supplied molecular system'],drawOptions=options)
        full.save(str(args.out/'figures'/f'input-{r["test_line"]:05d}.png'))
        dossier_index.append({'review_id':evidence['review_id'],'test_line':r['test_line'],'category':r['comparison']['category'],
            'network':r['sequence_id'],'reference_rank':r['target_rank'],'top1_count':r['top1_count'],
            'no_change':r['top1_no_change'],'cell':list(cell(r)),
            'path':str(path),'figure':str(args.out/evidence['figure']),
            'display':evidence['display'],'reference_edits':evidence['reference_edits']})
    (args.out/'dossier-index.json').write_text(json.dumps(dossier_index,indent=2)+'\n')
    print(json.dumps({'selection':str(args.out/'selection.json'),'coverage':manifest['coverage'],'lines':manifest['test_lines']},indent=2))


if __name__=='__main__':main()
