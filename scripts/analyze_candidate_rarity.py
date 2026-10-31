#!/usr/bin/env python3
"""Descriptive audit of saved candidates and full-training local-pattern support.

No generator/reranker fitting and no new or reserved test labels are accessed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase

import rarity_features as features
from summarize_saved_ranks import target_string

EXPERIMENT = Path('results/reranking/full-validation-test20k-20261029')
OLD = Path('results/reranking/failure-study/records.jsonl')
TRAIN = Path('data/flower_new_dataset/train.txt')
BANDS = ('0', '1-9', '10-99', '100-999', '1000+')
QUERIES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def coverage(rows):
    return {'rows': len(rows), 'top1_correct': sum(r['target_rank'] == 1 for r in rows),
            'top1_wrong': sum(r['target_rank'] != 1 for r in rows),
            'top10_recoverable': sum(r['recoverable'] for r in rows),
            'target_present_below_top10': sum(r['target_rank'] is not None and r['target_rank'] > 10 for r in rows),
            'target_absent_all64': sum(r['target_rank'] is None for r in rows),
            'empty_saved_pool': sum(r['top1'] is None for r in rows),
            'wrong_top1_target_tied': sum(r['target_rank'] != 1 and r['target_tied'] for r in rows)}


def prepare(out):
    if (out / 'cohort.jsonl').exists():
        return read_jsonl(out / 'cohort.jsonl')
    sources = [('previous_test', OLD), ('validation', EXPERIMENT / 'cache/val.jsonl')]
    sources += [('validation', p) for p in sorted((EXPERIMENT / 'inference/records').glob('val[0-9]*.jsonl'))]
    selection = json.loads((EXPERIMENT / 'inference/selection.json').read_text())
    assert sha(EXPERIMENT / 'cache/val.jsonl') == selection['cached_validation_records_sha256']
    for _, path in sources[2:]:
        receipt = json.loads(path.with_suffix('.manifest.json').read_text())
        assert receipt['status'] == 'complete' and sha(path) == receipt['records_sha256']
    rows, seen, queries = [], set(), defaultdict(set)
    with (out / 'cohort.jsonl.tmp').open('w') as stream:
        for split, path in sources:
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    identity = (split, row['source_row_id'])
                    assert identity not in seen
                    seen.add(identity)
                    rank = row['target_rank']
                    assert row['sample_count'] == sum(row['metrics']) == 64
                    assert (rank is None) == (row['metrics'][0] == 0)
                    top = row['candidates'][0] if row['candidates'] else None
                    assert (rank == 1) == bool(top and top['correct'])
                    recoverable = rank is not None and 1 < rank <= 10
                    family = features.signatures(row['reactant'], row['target'])
                    try:
                        top_family = features.signatures(row['reactant'], top['smiles']) if top else (None, None)
                    except ValueError:
                        top_family = (None, None)
                    queries[features.input_fingerprint(row['reactant'])].add(row['reactant'])
                    r = {'cohort': split, 'source_split': row['source_split'],
                         'source_row_id': row['source_row_id'], 'source_line': row['source_row_id'] + 1,
                         'sequence_id': row['sequence_id'], 'raw_sequence_id': row.get('raw_sequence_id', ''),
                         'input_key': hashlib.sha256(row['reactant'].encode()).hexdigest(),
                         'reactant': row['reactant'], 'target': row['target'],
                         'top1': top['smiles'] if top else None, 'target_rank': rank,
                         'target_count': row['metrics'][0], 'top1_count': top['count'] if top else 0,
                         'target_tied': row['target_tied_for_first'], 'recoverable': recoverable,
                         'target_no_op': row['target_no_op'], 'top1_no_op': row['top1_no_op'],
                         'atoms': row['reactant_properties']['atoms'], 'metrics': row['metrics'],
                         'target_family': family, 'top1_family': top_family,
                         'rerank_feature_supported': row['rerank_eligible']}
                    rows.append(r)
                    stream.write(json.dumps(r) + '\n')
                    if len(rows) % 10000 == 0:
                        print('Prepared saved rows', len(rows), flush=True)
    assert Counter(r['cohort'] for r in rows) == {'previous_test': 20152, 'validation': 21723}
    (out / 'cohort.jsonl.tmp').replace(out / 'cohort.jsonl')
    save(out / 'query-inputs.json', {k: sorted(v) for k, v in queries.items()})
    save(out / 'coverage.json', {split: coverage([r for r in rows if r['cohort'] == split])
                               for split in ('previous_test', 'validation')} | {'combined': coverage(rows)})
    save(out / 'sources.json', {'records_sha256': {str(p): sha(p) for _, p in sources},
                              'feature_version': features.VERSION,
                              'feature_code_sha256': sha(Path(features.__file__)),
                              'rdkit': rdBase.rdkitVersion, 'prepared_at': time.time(),
                              'scope': 'Official validation plus already inspected ordered test prefix only; descriptive pooled results.'})
    return rows


def initialize(query_path):
    global QUERIES
    QUERIES = {k: set(v) for k, v in json.loads(Path(query_path).read_text()).items()}


def train_batch(batch):
    counts = [Counter(), Counter()]
    witnesses, errors, tags = defaultdict(Counter), [], Counter()
    for index, raw in batch:
        try:
            pair, *metadata = raw.rstrip('\n').split('|')
            source, target = pair.split('>>')
            tag = metadata[-1] if metadata else ''
            tags['numeric_network' if tag.isdigit() else tag] += 1
            fs = features.signatures(source, target)
            for counter, family in zip(counts, fs):
                counter[family] += 1
            query = QUERIES.get(features.input_fingerprint(source))
            if query:
                normalized = target_string(source)
                if normalized in query:
                    key = hashlib.sha256(normalized.encode()).hexdigest()
                    witnesses[key][target_string(target)] += 1
        except Exception as error:
            errors.append({'row': index, 'error': type(error).__name__ + ': ' + str(error)})
    return counts, dict(witnesses), errors, tags, len(batch)


def batches(stream, size=512):
    values = enumerate(stream)
    while batch := list(itertools.islice(values, size)):
        yield batch


def index_training(out, workers):
    done = out / 'training-index.json'
    if done.exists():
        index = json.loads(done.read_text())
        assert index['feature_version'] == features.VERSION
        assert index['feature_code_sha256'] == sha(Path(features.__file__))
        return index
    counts, witnesses, errors, tags = [Counter(), Counter()], defaultdict(Counter), [], Counter()
    n, start, last = 0, time.time(), 0
    with TRAIN.open() as stream, Pool(workers, initialize, (str(out / 'query-inputs.json'),)) as pool:
        for local, matches, failures, local_tags, size in pool.imap(train_batch, batches(stream), chunksize=1):
            for counter, part in zip(counts, local):
                counter.update(part)
            for key, value in matches.items():
                witnesses[key].update(value)
            errors.extend(failures); tags.update(local_tags); n += size
            if n - last >= 50000:
                progress = {'rows': n, 'elapsed_seconds': time.time() - start,
                            'rows_per_second': n / (time.time() - start), 'parse_failures': len(errors)}
                save(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True); last = n
    assert n == 1933750
    result = {'training_rows': n, 'supported_rows': sum(counts[0].values()),
              'feature_version': features.VERSION, 'feature_code_sha256': sha(Path(features.__file__)),
              'train_sha256': sha(TRAIN), 'radius1': dict(counts[0]), 'radius2': dict(counts[1]),
              'known_targets_for_audited_inputs': dict(witnesses), 'parse_failures': errors,
              'source_tags': dict(tags), 'elapsed_seconds': time.time() - start,
              'frequency_definition': 'Number of official training rows with the same signed sparse Morgan count-fingerprint difference; repeated training rows are retained.'}
    save(done, result)
    print('Full training support index complete', n, flush=True)
    return result


def group_rows(rows):
    parent = list(range(len(rows))); seen = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, r in enumerate(rows):
        keys = [('input', r['input_key'])]
        if str(r['raw_sequence_id']).isdigit():
            keys.append(('network', r['source_split'], r['raw_sequence_id']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    for i, r in enumerate(rows):
        r['analysis_group'] = find(i)


def rarity_table(rows, radius, nonambiguous=False):
    selected = [r for r in rows if not r['target_no_op'] and
                (not nonambiguous or not r['multiple_recorded_targets'])]
    report = []
    for band in BANDS:
        group = [r for r in selected if features.support_band(r['target_support'][radius]) == band]
        info = coverage(group)
        info.update(band=band, unmatched_recoverable=sum(r['unmatched_recoverable'] for r in group))
        info.update(top1_error_rate=info['top1_wrong'] / len(group) if group else None,
                    pool_miss_rate=info['target_absent_all64'] / len(group) if group else None)
        report.append(info)
    return report


def association(rows, radius, nonambiguous=False):
    selected = [r for r in rows if not r['target_no_op'] and
                (not nonambiguous or not r['multiple_recorded_targets'])]
    grouped = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    strata = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for r in selected:
        rare = r['target_support'][radius] < 100
        offset = 0 if rare else 2
        for vector in (grouped[r['analysis_group']], strata[(r['atoms'] // 64, r['multiple_recorded_targets'])]):
            vector[offset] += r['target_rank'] != 1
            vector[offset + 1] += 1
    if not grouped:
        return {}
    values = np.asarray(list(grouped.values())); total = values.sum(axis=0)
    rates = [float(total[i] / total[i+1]) if total[i+1] else None for i in (0, 2)]
    rng = np.random.default_rng(20261030); differences = []
    for _ in range(2000):
        sample = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        if sample[1] and sample[3]:
            differences.append(sample[0] / sample[1] - sample[2] / sample[3])
    comparable = [v for v in strata.values() if v[1] and v[3]]
    n = sum(v[1] + v[3] for v in comparable)
    standardized = sum((v[1]+v[3]) * (v[0]/v[1]-v[2]/v[3]) for v in comparable) / n if n else None
    return {'rare_definition': '<100 training rows', 'rare_rows': int(total[1]), 'common_rows': int(total[3]),
            'rare_error_rate': rates[0], 'common_error_rate': rates[1],
            'difference_pp': 100 * (rates[0]-rates[1]) if None not in rates else None,
            'group_bootstrap_95ci_pp': [float(v*100) for v in np.quantile(differences, [.025,.975])] if differences else None,
            'analysis_groups': len(values), 'size_and_observed_ambiguity_standardized_difference_pp':
            100 * standardized if standardized is not None else None, 'standardization_comparable_rows': int(n)}


def report(out, rows, index):
    from diagnose_rank_failures import chemistry
    from targeted_reranker import describe
    known = defaultdict(set)
    for r in rows:
        known[r['input_key']].add(r['target'])
    for key, targets in index['known_targets_for_audited_inputs'].items():
        known[key].update(targets)
    group_rows(rows)
    for r in rows:
        r['multiple_recorded_targets'] = len(known[r['input_key']]) > 1
        r['top1_is_recorded_alternative'] = r['top1'] != r['target'] and r['top1'] in known[r['input_key']]
        r['unmatched_recoverable'] = r['recoverable'] and not r['top1_is_recorded_alternative']
        r['target_support'] = [index[name].get(family, 0) for name, family in zip(('radius1','radius2'), r['target_family'])]
        r['top1_support'] = [index[name].get(family, 0) if family is not None else None
                             for name, family in zip(('radius1','radius2'), r['top1_family'])]
        r['rarity_associated_shortlist'] = bool(r['unmatched_recoverable'] and not r['target_no_op']
            and not r['target_tied'] and r['target_support'][0] < 100 and r['top1_support'][0] is not None
            and r['top1_support'][0] >= max(100, 10 * max(1, r['target_support'][0])))
        if r['recoverable']:
            tc, pc = chemistry(r['target']), chemistry(r['top1'])
            r['difference_category'] = ('unsupported' if not tc['parseable'] or not pc['parseable'] else
                'same_heavy_bonds_H_charge' if tc.get('heavy_bond_signature') == pc.get('heavy_bond_signature') else
                'same_connectivity_other_changes' if tc.get('connectivity_signature') == pc.get('connectivity_signature') else
                'different_heavy_connectivity')
            pf, tf = describe(r['top1']).get('flags'), describe(r['target']).get('flags')
            r['top1_valence_flags'] = pf; r['target_valence_flags'] = tf
            r['top1_carbon_overcoordination_target_avoids'] = bool(pf and tf and pf['carbon_neighbors_gt4'] and not tf['carbon_neighbors_gt4'])
    result = {'coverage': {}, 'reranking_opportunity': {}, 'rarity': {}, 'associations': {},
              'training_rows': index['training_rows'], 'training_supported_rows': index['supported_rows'],
              'training_parse_failures': len(index['parse_failures']),
              'rarity_definition': index['frequency_definition'], 'training_source_tags': index['source_tags'],
              'scope': 'Descriptive 41,875 row observations; combined official validation and previously inspected ordered test prefix, not independent test accuracy.',
              'limitations': ['Associations do not identify causal training-data rarity.',
                  'Local fingerprints are hashed structural-change proxies, not named mechanism classes or proven chemistry.',
                  'Counts retain repeated training exposures; no-op references are excluded from rarity-rate comparisons.',
                  'Alternative labels are searched in training plus these two inspected cohorts only; missing witnesses do not prove invalidity.',
                  'Grouping preserves shared normalized inputs and source-specific numeric networks; only 2,000 descriptive group bootstrap resamples.',
                  'Standardization adjusts only atom-count bands and observed label multiplicity, leaving other confounders.',
                  'No candidate generation, model fitting, GPU rental, or reserved test-label inspection occurred.']}
    for split in ('previous_test', 'validation', 'combined'):
        cohort = rows if split == 'combined' else [r for r in rows if r['cohort'] == split]
        result['coverage'][split] = coverage(cohort)
        recovered = [r for r in cohort if r['recoverable']]
        unmatched = [r for r in recovered if r['unmatched_recoverable']]
        result['reranking_opportunity'][split] = {
            'recoverable': len(recovered), 'winner_is_recorded_alternative': sum(r['top1_is_recorded_alternative'] for r in recovered),
            'winner_not_observed_as_target': len(unmatched),
            'unmatched_difference_categories': dict(Counter(r['difference_category'] for r in unmatched)),
            'unmatched_baseline_no_op': sum(r['top1_no_op'] for r in unmatched),
            'unmatched_target_rank2': sum(r['target_rank'] == 2 for r in unmatched),
            'unmatched_target_tied_for_first': sum(r['target_tied'] for r in unmatched),
            'unmatched_feature_supported': sum(r['rerank_feature_supported'] for r in unmatched),
            'carbon_overcoordination_reference_avoids': sum(r['top1_carbon_overcoordination_target_avoids'] for r in unmatched),
            'rarity_associated_shortlist': sum(r['rarity_associated_shortlist'] for r in cohort),
            'rarity_shortlist_excluding_baseline_no_op': sum(r['rarity_associated_shortlist'] and not r['top1_no_op'] for r in cohort)}
        result['rarity'][split] = {f'radius{radius+1}': {'all_nontermination': rarity_table(cohort, radius),
                    'single_observed_target': rarity_table(cohort, radius, True)} for radius in (0,1)}
        result['associations'][split] = {f'radius{radius+1}': {'all_nontermination': association(cohort, radius),
                    'single_observed_target': association(cohort, radius, True)} for radius in (0,1)}
    a, b = ([r for r in rows if r['cohort'] == split] for split in ('previous_test','validation'))
    result['cross_cohort_shared_inputs'] = len({r['input_key'] for r in a} & {r['input_key'] for r in b})
    result['cross_cohort_shared_transitions'] = len({(r['input_key'],r['target']) for r in a} & {(r['input_key'],r['target']) for r in b})
    result['created_at'] = time.time()
    save(out / 'summary.json', result)
    for name, predicate in [('all-audited-rows', lambda r: True), ('recoverable-cases', lambda r:r['recoverable']),
            ('unmatched-recoverable-cases', lambda r:r['unmatched_recoverable']),
            ('rarity-associated-shortlist',lambda r:r['rarity_associated_shortlist']),
            ('missing-from-all-samples',lambda r:r['target_rank'] is None)]:
        with (out / f'{name}.jsonl').open('w') as f:
            for r in rows:
                if predicate(r):
                    f.write(json.dumps(r) + '\n')
    print(json.dumps({'coverage':result['coverage'],'opportunity':result['reranking_opportunity'],
                      'combined_associations':result['associations']['combined']},indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--stage', choices=('prepare','index','report','all'), default='all')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = prepare(args.out)
    if args.stage == 'prepare':
        return
    index = index_training(args.out, args.workers)
    if args.stage != 'index':
        report(args.out, rows, index)


if __name__ == '__main__':
    main()
