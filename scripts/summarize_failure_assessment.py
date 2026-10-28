#!/usr/bin/env python3
"""Describe frozen reranker decisions; never retune or change the assessment."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(outcomes):
    transitions = Counter(row['transition'] for row in outcomes)
    net = sum(int(row['correct']) - int(row['baseline_correct']) for row in outcomes)
    return dict(rows=len(outcomes), baseline_correct=sum(row['baseline_correct'] for row in outcomes),
                corrected=transitions['corrected'], harmed=transitions['harmed'], net_correct=net,
                net_percentage_points=100*net/len(outcomes) if outcomes else None,
                changed_wrong_to_wrong=sum(row['changed_wrong_to_wrong'] for row in outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reserved', type=Path, required=True)
    args = parser.parse_args()
    study = args.study
    rows = read_rows(study/'records.jsonl')
    by_id = {row['row_id']: row for row in rows}
    choices = read_rows(study/'assessment/choices.jsonl')
    comparison = json.loads((study/'assessment/comparison.json').read_text())
    receipt = json.loads((study/'assessment/artifact-receipt.json').read_text())
    assert all(digest(study/'assessment'/name) == sha for name, sha in receipt.items())
    splits = json.loads((study/'learner/splits.json').read_text())
    assert set(row['row_id'] for row in choices) == set(splits['row_ids']['assessment'])
    reserved = json.loads(args.reserved.read_text())
    separation = {}
    for key in ('source_row_id', 'split_group', 'reactant_key', 'sequence_id'):
        generic = {'', 'PM', 'RS', 'RC', 'PC'} if key == 'sequence_id' else set()
        overlap = ({row[key] for row in rows} - generic) & ({row[key] for row in reserved} - generic)
        assert not overlap, f'Reserved evaluation shares {key} with diagnosed prefix'
        separation[key] = len(overlap)
    result = {'status': 'exploratory_post_assessment_description_not_retuning',
              'assessment_rows': len(choices), 'reserved_rows': len(reserved),
              'reserved_overlap_counts': separation, 'methods': {},
              'source_sha256': {'records': digest(study/'records.jsonl'),
                                'choices': digest(study/'assessment/choices.jsonl'),
                                'reserved': digest(args.reserved)}}
    changed = []
    for method in ('simple', 'transform_fingerprint'):
        cohorts, groups = defaultdict(list), defaultdict(list)
        rescue_patterns, harm_patterns = Counter(), Counter()
        for choice in choices:
            row, outcome = by_id[choice['row_id']], choice['methods'][method]
            cohorts['all'].append(outcome)
            cohorts['multiple_recorded_targets' if row['reactant_has_multiple_recorded_targets']
                    else 'one_observed_target'].append(outcome)
            cohorts['reference_no_change' if row['target_no_op'] else 'reference_changes'].append(outcome)
            cohorts['baseline_correct' if row['top1_correct'] else 'baseline_wrong'].append(outcome)
            groups[row['split_group']].append(outcome)
            if outcome['transition'] == 'corrected':
                rescue_patterns.update(key for key, value in row.get('diagnostic_flags', {}).items() if value)
            elif outcome['transition'] == 'harmed':
                harm_patterns['reference_no_change'] += row['target_no_op']
                harm_patterns['input_has_multiple_recorded_targets'] += row['reactant_has_multiple_recorded_targets']
            if outcome['changed']:
                changed.append({'method': method, 'transition': outcome['transition'],
                                'changed_wrong_to_wrong': outcome['changed_wrong_to_wrong'],
                                'selected_rank': outcome['selected_rank'],
                                'selected_smiles': outcome['selected_smiles'],
                                'source_line_1based': row['source_row_id'] + 1,
                                'record': row})
        summaries = {key: counts(values) for key, values in cohorts.items()}
        for key in ('rows', 'baseline_correct', 'corrected', 'harmed', 'net_correct', 'changed_wrong_to_wrong'):
            assert summaries['all'][key] == comparison['methods'][method][key]
        group_net = {key: counts(values)['net_correct'] for key, values in groups.items()}
        result['methods'][method] = dict(cohorts=summaries,
            group_net_counts={'positive': sum(n > 0 for n in group_net.values()),
                              'zero': sum(n == 0 for n in group_net.values()),
                              'negative': sum(n < 0 for n in group_net.values())},
            corrected_case_patterns=dict(rescue_patterns), newly_wrong_patterns=dict(harm_patterns))
    (study/'assessment/diagnostic-breakdown.json').write_text(json.dumps(result, indent=2)+'\n')
    with (study/'assessment/changed-cases.jsonl').open('w') as stream:
        for row in changed:
            stream.write(json.dumps(row)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
