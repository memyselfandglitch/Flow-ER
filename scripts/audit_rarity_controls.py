#!/usr/bin/env python3
"""Post hoc controls for rarity/validity hypotheses; no scoring or fitting."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

from rdkit import Chem
import rarity_features as features
from analyze_candidate_rarity import EXPERIMENT, OLD, read_jsonl, save

PARSER = Chem.SmilesParserParams()
PARSER.removeHs = False
PARSER.sanitize = False


@lru_cache(maxsize=8192)
def flags(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None:
        return None
    return {
        'carbon_neighbors_gt4': any(a.GetAtomicNum() == 6 and a.GetDegree() > 4 for a in mol.GetAtoms()),
        'carbon_aluminum_charge_pair': (
            any(a.GetAtomicNum() == 6 and a.GetFormalCharge() >= 2 for a in mol.GetAtoms())
            and any(a.GetAtomicNum() == 13 and a.GetFormalCharge() <= -2 for a in mol.GetAtoms()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); out = args.out
    rows = read_jsonl(out / 'cohort.jsonl')
    index = json.loads((out / 'training-index.json').read_text())
    support = index['radius1']
    known = defaultdict(set)
    for r in rows:
        known[r['input_key']].add(r['target'])
    for key, targets in index['known_targets_for_audited_inputs'].items():
        known[key].update(targets)
    for r in rows:
        r['top1_is_recorded_alternative'] = r['top1'] != r['target'] and r['top1'] in known[r['input_key']]
        r['unmatched_recoverable'] = r['recoverable'] and not r['top1_is_recorded_alternative']
        r['target_support'] = [index[name].get(family, 0) for name, family in zip(('radius1','radius2'),r['target_family'])]
        r['top1_support'] = [index[name].get(family, 0) if family is not None else None
                             for name, family in zip(('radius1','radius2'),r['top1_family'])]
    lookup = {(r['cohort'], r['source_row_id']): r for r in rows}
    sources = [('previous_test', OLD), ('validation', EXPERIMENT / 'cache/val.jsonl')]
    sources += [('validation', q) for q in sorted((EXPERIMENT / 'inference/records').glob('val[0-9]*.jsonl'))]
    controls = {s: {'rare_alternative_gate': Counter(), 'rare_alternative_gate_excluding_no_op_winner': Counter(),
                    'carbon_neighbors_gt4': Counter(), 'carbon_aluminum_charge_pair': Counter(),
                    'missing_pool': Counter()} for s in ('previous_test', 'validation', 'combined')}
    flagged = []; count = 0
    for cohort, path in sources:
        with path.open() as stream:
            for line in stream:
                original = json.loads(line)
                r = lookup[(cohort, original['source_row_id'])]
                pf, tf = flags(r['top1']), flags(r['target'])
                rare_candidates = []
                if r['top1_support'][0] is not None and r['top1_support'][0] >= 100:
                    for c in original['candidates'][1:]:
                        try:
                            family = features.signatures(r['reactant'], c['smiles'])[0]
                            n = support.get(family, 0)
                        except ValueError:
                            continue
                        if n < 100 and r['top1_support'][0] >= 10 * max(1, n):
                            rare_candidates.append({'rank': c['rank'], 'support': n, 'correct': c['correct'], 'count': c['count']})
                row_flags = []
                for split in (cohort, 'combined'):
                    result = controls[split]
                    if rare_candidates:
                        gate_names = ['rare_alternative_gate']
                        if not r['top1_no_op']:
                            gate_names.append('rare_alternative_gate_excluding_no_op_winner')
                        for key in gate_names:
                            counts = result[key]
                            counts['flagged_rows'] += 1
                            counts['baseline_correct'] += r['target_rank'] == 1
                            counts['baseline_wrong'] += r['target_rank'] != 1
                            counts['target_among_flagged_alternatives'] += any(c['correct'] for c in rare_candidates)
                            counts['unmatched_target_among_flagged_alternatives'] += (
                                not r['top1_is_recorded_alternative'] and any(c['correct'] for c in rare_candidates))
                            counts['multiple_flagged_alternatives'] += len(rare_candidates) > 1
                    for key in ('carbon_neighbors_gt4', 'carbon_aluminum_charge_pair'):
                        counts = result[key]
                        counts['reference_flagged'] += bool(tf and tf[key])
                        counts['unsupported_top1'] += pf is None
                        if pf and pf[key]:
                            counts['baseline_flagged'] += 1
                            counts['baseline_correct'] += r['target_rank'] == 1
                            counts['baseline_wrong'] += r['target_rank'] != 1
                            counts['recoverable'] += r['recoverable']
                            counts['unmatched_recoverable'] += r['unmatched_recoverable']
                            counts['target_in_top10_and_avoids_flag'] += r['recoverable'] and bool(tf and not tf[key])
                            counts['target_absent_all64'] += r['target_rank'] is None
                            if split == cohort:
                                row_flags.append(key)
                    if r['target_rank'] is None:
                        counts = result['missing_pool']
                        counts['rows'] += 1
                        counts['target_nontermination'] += not r['target_no_op']
                        counts['rare_nontermination_r1'] += not r['target_no_op'] and r['target_support'][0] < 100
                        counts['rare_nontermination_r2'] += not r['target_no_op'] and r['target_support'][1] < 100
                        counts['at_least_half_conserved_exception_draws'] += r['metrics'][3] >= 32
                        counts['all64_conserved_exception_draws'] += r['metrics'][3] == 64
                        counts['at_least_half_unsaved_draws'] += sum(r['metrics'][2:]) >= 32
                        for i, n in enumerate(r['metrics']):
                            counts[f'draw_bucket_{i}'] += n
                if row_flags or rare_candidates:
                    flagged.append({'cohort': cohort, 'source_line': r['source_line'], 'target_rank': r['target_rank'],
                        'flags': row_flags, 'rare_alternatives': rare_candidates, 'top1_support': r['top1_support'],
                        'top1_no_op': r['top1_no_op'], 'top1_is_recorded_alternative': r['top1_is_recorded_alternative']})
                count += 1
                if count % 10000 == 0:
                    print('Checked controls', count, flush=True)
    assert count == 41875
    save(out / 'controls.json', {
        'definition': 'A label-blind gate: winner support >=100; at least one lower-ranked top-10 candidate has <100 support and at least 10-fold lower support than winner. This counts flags, not a fitted or validated selector.',
        'candidate_rarity_radius': 1, 'rows_checked': count, 'cohorts': controls,
        'cautions': ['Rare alternatives can be incorrect; an activation flag does not identify a replacement.',
                     'Carbon-neighbor and C/Al charge checks are specific structural flags, not a full chemical validity test.',
                     'Exception buckets do not identify the failed conversion operation; preconversion states were not saved.']})
    with (out / 'control-flags.jsonl').open('w') as stream:
        for r in flagged:
            stream.write(json.dumps(r) + '\n')
    print(json.dumps(controls, indent=2), flush=True)


if __name__ == '__main__':
    main()
