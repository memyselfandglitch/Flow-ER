#!/usr/bin/env python3
"""Check two review-derived flags against cached correct controls.

Post hoc descriptive audit only. No candidate reordering or fitting.
"""
from collections import Counter
import json
from pathlib import Path
from rdkit import Chem
from rerank_analysis import PARSER


def flags(smiles):
    if not smiles:
        return set()
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None:
        return {'unparseable'}
    result = set()
    # Five explicit neighbors is independent of aromatic bond-order conventions.
    if any(a.GetAtomicNum() == 6 and a.GetDegree() > 4 for a in mol.GetAtoms()):
        result.add('carbon_more_than_four_explicit_neighbors')
    charges = {(a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms()}
    if ('C', 2) in charges and ('Al', -2) in charges:
        result.add('paired_carbon_plus2_aluminium_minus2')
    return result


rows = []
denom = Counter()
for line in Path('results/reranking/failure-study/records.jsonl').open():
    d = json.loads(line)
    cohort = 'baseline_correct' if d['top1_correct'] else 'baseline_wrong'
    denom[cohort] += 1
    top = flags(d['candidates'][0]['smiles'] if d['candidates'] else None)
    ref = flags(d['target'])
    if top or ref:
        rows.append({
            'test_line':d['row_id']+1, 'cohort':cohort,
            'top1_flags':sorted(top), 'reference_flags':sorted(ref),
            'reference_rank':d['target_rank'], 'sequence_id':d['sequence_id'],
        })
summary = {
    'scope':'Post hoc diagnostic audit of two review-derived flags; not held-out performance, a fitted rule, or an energy test',
    'source':'results/reranking/failure-study/records.jsonl',
    'denominators':dict(denom), 'flags':{},
}
for flag in ('carbon_more_than_four_explicit_neighbors', 'paired_carbon_plus2_aluminium_minus2'):
    flagged = [r for r in rows if flag in r['top1_flags']]
    summary['flags'][flag] = {
        'top1_flagged':len(flagged),
        'baseline_correct':sum(r['cohort']=='baseline_correct' for r in flagged),
        'baseline_wrong':sum(r['cohort']=='baseline_wrong' for r in flagged),
        'reference_in_top10':sum(r['reference_rank'] is not None and r['reference_rank']<=10 for r in flagged),
        'reference_flagged_any_row':sum(flag in r['reference_flags'] for r in rows),
        'test_lines':[r['test_line'] for r in flagged],
    }
out = Path('results/stratified-review-50/flag-control-audit.json')
out.write_text(json.dumps(dict(summary, cases=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2))
