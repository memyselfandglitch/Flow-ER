#!/usr/bin/env python3
"""Audit every saved ranking failure without silently filtering any candidate."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import ast

import numpy as np
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdMolDescriptors
from rerank_analysis import FEATURES,FP,PARSER
from summarize_saved_ranks import target_string
from prepare_rerank_pilot import joined_groups

RDLogger.DisableLog('rdApp.*')


@lru_cache(maxsize=100000)
def reference(smiles):
    return target_string(smiles)


def graph_signature(mol,ignore_bond_order=False):
    heavy=Chem.RWMol(mol)
    for index in range(heavy.GetNumAtoms()-1,-1,-1):
        if heavy.GetAtomWithIdx(index).GetAtomicNum()==1: heavy.RemoveAtom(index)
    for atom in heavy.GetAtoms():
        atom.SetAtomMapNum(0);atom.SetFormalCharge(0);atom.SetNumRadicalElectrons(0)
        atom.SetNumExplicitHs(0);atom.SetNoImplicit(True)
        if ignore_bond_order: atom.SetIsAromatic(False)
    if ignore_bond_order:
        for bond in heavy.GetBonds(): bond.SetBondType(Chem.BondType.SINGLE);bond.SetIsAromatic(False)
    return Chem.MolToSmiles(heavy,isomericSmiles=False)


@lru_cache(maxsize=100000)
def chemistry(smiles):
    mol=Chem.MolFromSmiles(smiles,PARSER)
    if mol is None: return {'parseable':False,'reason':'saved_smiles_cannot_be_reparsed'}
    charges=[atom.GetFormalCharge() for atom in mol.GetAtoms()]
    properties={'formula':rdMolDescriptors.CalcMolFormula(mol),'charge':sum(charges),
                'abs_charge':sum(abs(q) for q in charges),'charged_atoms':sum(q!=0 for q in charges),
                'fragments':len(Chem.GetMolFrags(mol)),'radicals':sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
                'heavy_atoms':mol.GetNumHeavyAtoms(),'atoms':mol.GetNumAtoms(),
                'rings':rdMolDescriptors.CalcNumRings(mol),'aromatic_atoms':sum(a.GetIsAromatic() for a in mol.GetAtoms())}
    result={'parseable':True,'properties':properties,'fingerprint':FP.GetFingerprint(mol)}
    for key,ignore in [('heavy_bond_signature',False),('connectivity_signature',True)]:
        try: result[key]=graph_signature(mol,ignore)
        except Exception: result[key]=None
    return result


def numeric_features(reactant,candidate,count,samples,valid):
    rp=reactant['properties'];cp=candidate['properties']
    feature=[math.log((count+.5)/(samples+1)),count/max(1,valid),
             DataStructs.TanimotoSimilarity(reactant['fingerprint'],candidate['fingerprint']),0.]
    feature += [cp[k]-rp[k] for k in ['abs_charge','charged_atoms','fragments','radicals','rings','aromatic_atoms']]
    feature += [cp[k] for k in ['abs_charge','fragments','radicals']]
    assert len(feature)==len(FEATURES)
    return feature


def analyze_saved(truth,line,index,meta,k=10):
    fields=truth.rstrip('\n').split('|');raw_reactant,raw_target=fields[0].split('>>')
    reactant,target=reference(raw_reactant),reference(raw_target)
    assert hashlib.sha256(reactant.encode()).hexdigest()==meta['reactant_key']
    assert hashlib.sha256(truth.encode()).hexdigest()==meta['source_line_sha256']
    m,asym,raw=line.rstrip('\n').split('|',2);metrics=ast.literal_eval(m);raw=ast.literal_eval(raw)
    assert len(metrics)==5 and sum(metrics)==64 and all(isinstance(x,int) and x>=0 for x in metrics)
    assert all(flag is True and isinstance(n,int) and n>0 for smi,n,flag in raw)
    assert len({smi for smi,_,_ in raw})==len(raw)
    assert sum(n for _,n,_ in raw)==metrics[0]+metrics[1]
    counts={smi:n for smi,n,_ in raw};assert counts.get(target,0)==metrics[0],f'Alignment mismatch at {index}'
    ordered=sorted(raw,key=lambda item:item[1],reverse=True)
    rank=next((j+1 for j,(smi,_,_) in enumerate(ordered) if smi==target),None)
    rc=chemistry(reactant);assert rc['parseable']
    candidates=[];unsupported=[]
    for j,(smi,n,_) in enumerate(ordered[:k],1):
        cc=chemistry(smi)
        if cc['parseable']:
            features=numeric_features(rc,cc,n,64,sum(counts.values()));features[3]=float(smi==reactant)
        else:
            features=[math.log((n+.5)/65),n/max(1,sum(counts.values()))]+[0.]*(len(FEATURES)-2)
            unsupported.append(j)
        candidates.append({'rank':j,'smiles':smi,'count':n,'correct':smi==target,
                           'no_op':smi==reactant,'parseable':cc['parseable'],'properties':cc.get('properties'),
                           'features':features,'feature_status':'ok' if cc['parseable'] else cc['reason']})
    raw_sequence=fields[-1] if len(fields)>1 else ''
    top=ordered[0] if ordered else None
    probabilities=[n/sum(counts.values()) for _,n,_ in ordered] if ordered else []
    row={**meta,'row_id':index,'source_row_id':index,'source_split':'test','reaction':fields[0],
         'raw_sequence_id':raw_sequence,'reactant':reactant,'target':target,'sample_count':64,'metrics':metrics,
         'reactant_properties':rc['properties'],'target_rank':rank,'top1_correct':rank==1,
         'valid_conserved_samples':sum(counts.values()),'distinct_saved_candidates':len(ordered),
         'candidates':candidates,'rerank_eligible':not unsupported,'unsupported_candidate_ranks':unsupported,
         'target_tied_for_first':bool(top and rank and counts[target]==top[1]),
         'top1_target_count_gap':top[1]-counts[target] if top and rank else None,
         'top1_top2_count_gap':ordered[0][1]-ordered[1][1] if len(ordered)>1 else None,
         'top1_sample_fraction':top[1]/64 if top else 0.,
         'candidate_entropy':-sum(p*math.log(p) for p in probabilities),
         'top1_no_op':bool(top and top[0]==reactant),'target_no_op':target==reactant}
    if len(ordered)>1:
        first,second=chemistry(ordered[0][0]),chemistry(ordered[1][0])
        row['top_two_same_heavy_connectivity']=bool(first.get('connectivity_signature') is not None and first.get('connectivity_signature')==second.get('connectivity_signature'))
    else: row['top_two_same_heavy_connectivity']=None
    return row


def diagnose(row,targets,sequence_targets):
    top=row['candidates'][0];truth=chemistry(row['target']);pred=chemistry(top['smiles'])
    known=targets[row['reactant']]
    flags={'top1_is_other_recorded_target':top['smiles'] in known,
           'top1_is_other_target_in_same_pathway':top['smiles'] in sequence_targets[(row['reactant'],row['sequence_id'])],
           'reactant_has_multiple_recorded_targets':len(known)>1,
           'top1_no_op':row['top1_no_op'],'target_is_termination':row['target_no_op'],
           'tied_for_highest_frequency':row['target_tied_for_first'],
           'frequency_gap_le4':row['top1_target_count_gap']<=4,
           'frequency_gap_le8':row['top1_target_count_gap']<=8,
           'unsupported_candidate_in_top10':not row['rerank_eligible']}
    if truth['parseable'] and pred['parseable']:
        same_conn=truth.get('connectivity_signature') is not None and truth.get('connectivity_signature')==pred.get('connectivity_signature')
        same_bonds=truth.get('heavy_bond_signature') is not None and truth.get('heavy_bond_signature')==pred.get('heavy_bond_signature')
        flags.update(same_heavy_connectivity=same_conn,same_heavy_bonds=same_bonds,
                     bond_order_change_same_connectivity=same_conn and not same_bonds,
                     different_heavy_connectivity=not same_conn,
                     different_fragment_count=truth['properties']['fragments']!=pred['properties']['fragments'],
                     different_charge_separation=truth['properties']['abs_charge']!=pred['properties']['abs_charge'],
                     different_radical_count=truth['properties']['radicals']!=pred['properties']['radicals'])
    row['diagnostic_flags']=flags
    row['other_recorded_target_counts']=dict(known)
    return row


def cohort_summary(rows):
    multi=[r for r in rows if len(r['candidates'])>=2]
    return {'rows':len(rows),'multiple_candidate_rows':len(multi),
            'mean_atoms':float(np.mean([r['reactant_properties']['atoms'] for r in rows])) if rows else None,
            'mean_top1_sample_fraction':float(np.mean([r['top1_sample_fraction'] for r in rows])) if rows else None,
            'median_candidate_count':float(np.median([r['distinct_saved_candidates'] for r in rows])) if rows else None,
            'top1_no_op':sum(r['top1_no_op'] for r in rows),'target_no_op':sum(r['target_no_op'] for r in rows),
            'reactant_multiple_targets':sum(r['reactant_has_multiple_recorded_targets'] for r in rows),
            'among_multiple_candidates':{'rows':len(multi),
                'mean_top1_sample_fraction':float(np.mean([r['top1_sample_fraction'] for r in multi])) if multi else None,
                'median_top1_top2_gap':float(np.median([r['top1_top2_count_gap'] for r in multi])) if multi else None,
                'gap_le4':sum(r['top1_top2_count_gap']<=4 for r in multi),
                'top_two_same_heavy_connectivity':sum(bool(r['top_two_same_heavy_connectivity']) for r in multi)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',type=Path,required=True);p.add_argument('--pred',type=Path,required=True)
    p.add_argument('--index-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'records.jsonl').exists(),'Preserve existing completed extraction; use another output directory'
    metadata=[]
    for part in ['train','val','test']:
        metadata.extend(json.loads(line) for line in (a.index_root/f'{part}-input-index.jsonl').open())
    joined_groups(metadata)
    test_meta={r['source_row_id']:r for r in metadata if r['source_split']=='test'}
    rows=[];targets=defaultdict(Counter);seq_targets=defaultdict(set);sha=hashlib.sha256()
    with a.gt.open() as gt,a.pred.open('rb') as predictions:
        for index,raw in enumerate(predictions):
            assert raw.endswith(b'\n')
            row=analyze_saved(next(gt),raw.decode(),index,test_meta[index])
            rows.append(row);targets[row['reactant']][row['target']]+=1
            seq_targets[(row['reactant'],row['sequence_id'])].add(row['target']);sha.update(raw)
            if len(rows)%1000==0: print('Audited',len(rows),'rows',flush=True)
    failures=[]
    for row in rows:
        row['reactant_has_multiple_recorded_targets']=len(targets[row['reactant']])>1
        if row['target_rank'] is not None and 1<row['target_rank']<=10:
            failures.append(diagnose(row,targets,seq_targets))
    for name,cohort in [('records',rows),('top1_wrong_top10_correct',failures)]:
        with (a.out/f'{name}.jsonl').open('w') as stream:
            for row in cohort: stream.write(json.dumps(row)+'\n')
    patterns=Counter(key for row in failures for key,value in row['diagnostic_flags'].items() if value)
    correct=[r for r in rows if r['top1_correct']]
    error_ranks=Counter(r['target_rank'] for r in failures)
    gaps=[r['top1_target_count_gap'] for r in failures]
    summary={'status':'complete_diagnostic_prefix','rows':len(rows),'failure_rows':len(failures),
             'top1_correct':len(correct),'top1_wrong':len(rows)-len(correct),'patterns_overlap':True,
             'patterns':dict(patterns),'target_rank_in_failure_cases':dict(sorted(error_ranks.items())),
             'frequency_gap_top1_minus_target':{'median':float(np.median(gaps)),'p25':float(np.quantile(gaps,.25)),'p75':float(np.quantile(gaps,.75))},
             'cohorts':{'rescued_by_top10':cohort_summary(failures),'baseline_correct':cohort_summary(correct)},
             'unsupported_feature_rows':sum(not r['rerank_eligible'] for r in rows),
             'unsupported_candidate_entries':sum(len(r['unsupported_candidate_ranks']) for r in rows),
             'distinct_reactants':len(targets),'reactants_with_multiple_recorded_targets':sum(len(v)>1 for v in targets.values()),
             'empirical_best_fixed_target_per_reactant_accuracy':sum(max(v.values()) for v in targets.values())/len(rows),
             'feature_names':FEATURES,'prediction_sha256':sha.hexdigest(),
             'limitations':['Observed patterns are diagnostic associations, not proven chemical causes.',
                           'All cases in this ordered prefix were inspected; any development assessment here is exploratory.',
                           'A lookup using recorded targets is diagnostic only and is excluded from learned scoring features.',
                           'Unsupported candidate strings remain in rankings; feature reranking must fall back to frequency for those rows.']}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
