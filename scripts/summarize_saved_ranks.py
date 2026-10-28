#!/usr/bin/env python3
"""Audit ranks in saved FlowER candidate lists without changing candidate sets."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
PARSER=Chem.SmilesParserParams();PARSER.removeHs=False;PARSER.sanitize=True


def target_string(mapped):
    mol=Chem.MolFromSmiles(mapped,PARSER)
    if mol is None: raise ValueError('Invalid reference molecule')
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    result=Chem.MolToSmiles(mol,isomericSmiles=False,allHsExplicit=True)
    # Match the evaluation writer's explicit-H SMILES roundtrip.
    mol=Chem.MolFromSmiles(result,PARSER)
    if mol is None: raise ValueError('Reference does not roundtrip')
    return Chem.MolToSmiles(mol,isomericSmiles=False,allHsExplicit=True)


def summarize(gt_path,pred_path):
    counts=Counter();sha=hashlib.sha256();rank_hist=Counter();ties=Counter()
    with gt_path.open() as gt,pred_path.open('rb') as pred:
        for i,raw in enumerate(pred):
            assert raw.endswith(b'\n'),f'Incomplete prediction row {i}'
            truth=next(gt,None);assert truth is not None,'Extra prediction rows'
            reference=target_string(truth.split('|')[0].split('>>')[1])
            metrics,_,serialized=raw.decode().rstrip('\n').split('|',2)
            metrics=ast.literal_eval(metrics);candidates=ast.literal_eval(serialized)
            assert len(metrics)==5 and all(isinstance(x,int) and x>=0 for x in metrics)
            assert sum(metrics)==64,f'Incomplete sampling at row {i}'
            assert all(flag is True and isinstance(n,int) and n>0 for smi,n,flag in candidates)
            assert len({smi for smi,_,_ in candidates})==len(candidates),'Duplicate saved keys'
            assert sum(n for _,n,_ in candidates)==metrics[0]+metrics[1],f'Count mismatch {i}'
            matches=[(smi,n) for smi,n,_ in candidates if smi==reference]
            assert sum(n for _,n in matches)==metrics[0],f'Target/tally alignment mismatch at row {i}'
            ordered=sorted(candidates,key=lambda c:c[1],reverse=True)
            rank=next((j+1 for j,(smi,_,_) in enumerate(ordered) if smi==reference),None)
            count=matches[0][1] if matches else 0
            best=1+sum(n>count for _,n,_ in ordered) if matches else None
            worst=sum(n>=count for _,n,_ in ordered) if matches else None
            counts['rows']+=1;counts['top1_correct']+=rank==1
            counts['top1_wrong']+=rank!=1;counts['no_saved_candidates']+=not ordered
            counts['recorded_target_absent_from_all_samples']+=rank is None
            counts['top1_wrong_but_target_tied_for_first']+=bool(rank and rank>1 and best==1)
            rank_hist[str(rank) if rank else 'absent']+=1
            for k in [3,5,10]:
                hit=rank is not None and rank<=k
                counts[f'top{k}_correct']+=hit
                counts[f'top{k}_correct_top1_wrong']+=hit and rank!=1
                counts[f'top{k}_miss']+=not hit
                counts[f'top{k}_strict_frequency_errors']+=bool(hit and rank!=1 and best>1)
                ties[f'top{k}_target_at_boundary']+=bool(best and best<=k<worst)
            sha.update(raw)
    n=counts['rows']
    return {'predictions_analyzed':n,'prediction_sha256':sha.hexdigest(),
            'unit':'one supplied elementary-step reactant/intermediate -> recorded next state',
            'ranking':'saved distinct candidate states sorted by sample count; original stable order breaks ties',
            'candidate_policy':'Preserve every saved candidate and its count; no chemical reparsing/filtering of candidate strings',
            'reference_policy':'Matches eval_multiGPU.py explicit-H/no-map/no-stereo roundtrip; target counts checked against every saved metric',
            'samples_per_step':64,'counts':dict(counts),'rank_histogram':dict(rank_hist),'tie_boundary_counts':dict(ties),
            'accuracy':{f'top{k}':counts[f'top{k}_correct']/n for k in [1,3,5,10]},
            'scope':'Ordered completed prefix only; not the full test benchmark and not pathway/beam-search accuracy'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',type=Path,required=True);p.add_argument('--pred',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=summarize(a.gt,a.pred);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
