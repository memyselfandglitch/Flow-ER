#!/usr/bin/env python3
"""Fit a new reranker on all official validation; evaluate a frozen test sample.

Model settings are selected by grouped validation-only cross-validation. Old
candidate predictions may be reused, but no old trained reranker is loaded.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np

import targeted_reranker as core
from pilot_reranker import outcome_rows, summarize, tuning_key, cluster_bootstrap
from run_targeted_inference import sha256

VERSION = 'full-validation-new-reranker-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def implementation():
    names = ('full_validation_reranker.py', 'targeted_reranker.py',
             'pilot_reranker.py', 'rerank_analysis.py')
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def experiment(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, digest in protocol['manifest_sha256'].items():
        require(sha256(root / name) == digest, 'Frozen cohort manifest changed')
    return protocol


def validate_membership(rows, metadata, split):
    require(len(rows) == len(metadata), f'Incomplete {split} records')
    expected = {r['source_row_id']: r for r in metadata}
    seen = set()
    for row in rows:
        identity = row['source_row_id']
        require(identity in expected and identity not in seen, 'Duplicate or foreign source row')
        seen.add(identity)
        meta = expected[identity]
        for field in ('source_split', 'sequence_id', 'reactant_key', 'split_group', 'source_line_sha256'):
            require(row[field] == meta[field], f'Record membership mismatch: {field}')
        require(row['sample_count'] == 64, 'Every input must have exactly 64 samples')
        row['row_id'] = identity
        row['selection_metadata'] = meta
        row['cv_fold'] = meta.get('cv_fold')
        row['candidate_origin'] = meta['candidate_origin']
    require(seen == set(expected), 'Missing original source rows')
    rows.sort(key=lambda r: r['source_row_id'])
    return core.validate_records(rows, split)


def load_complete(root, split):
    protocol = experiment(root)
    name = 'validation-rows.json' if split == 'val' else 'test-rows.json'
    metadata = json.loads((root / name).read_text())
    selection = json.loads((root / 'inference' / 'selection.json').read_text())
    require(selection['protocol_sha256'] == sha256(root / 'protocol.json'), 'Inference protocol differs')
    rows, hashes = [], {}
    if split == 'val':
        cache = root / 'cache' / 'val.jsonl'
        receipt = json.loads((root / 'cache' / 'manifest.json').read_text())
        digest = sha256(cache)
        require(digest == receipt['records_sha256'] == selection['cached_validation_records_sha256'],
                'Cached validation candidates changed')
        imported = [json.loads(line) for line in cache.open()]
        require(len(imported) == receipt['rows'] == protocol['reused_validation_rows'], 'Incomplete cache')
        rows.extend(imported); hashes[str(cache.relative_to(root))] = digest
    for part, spec in selection['parts'].items():
        if spec['source_split'] != split:
            continue
        path = root / 'inference' / 'records' / f'{part}.jsonl'
        receipt_path = path.with_suffix('.manifest.json')
        require(path.exists() and receipt_path.exists(), f'Inference incomplete: {part}')
        receipt = json.loads(receipt_path.read_text())
        digest = sha256(path)
        require(receipt['status'] == 'complete' and receipt['rows'] == spec['rows'], 'Incomplete part receipt')
        require(digest == receipt['records_sha256'] and receipt['input_sha256'] == spec['inputs_sha256'] and
                receipt['row_manifest_sha256'] == spec['row_manifest_sha256'], 'Part provenance differs')
        imported = [json.loads(line) for line in path.open()]
        require(len(imported) == spec['rows'], 'Part length differs')
        rows.extend(imported); hashes[str(path.relative_to(root))] = digest
    return validate_membership(rows, metadata, split), hashes


def configs():
    return [{'C': None, 'alpha': 0., 'margin': 0.}] + [
        {'C': c, 'alpha': alpha, 'margin': margin}
        for c, alpha, margin in itertools.product(core.CS, core.ALPHAS, core.MARGINS)]


def cross_validate(rows, matrices, folds=5):
    grid = configs()
    # Collect only outcome counters, not repeated large SMILES tables.
    counts = [Counter() for _ in grid]
    rejected = set()
    training = []
    for fold in range(folds):
        fit_ids = [i for i, row in enumerate(rows) if row['cv_fold'] != fold]
        tune_ids = [i for i, row in enumerate(rows) if row['cv_fold'] == fold]
        require(fit_ids and tune_ids, 'Every grouped CV fold must be nonempty')
        fit_rows, tune_rows = ([rows[i] for i in indices] for indices in (fit_ids, tune_ids))
        core.assert_disjoint(core.signature(fit_rows), core.signature(tune_rows), f'CV fold {fold}')
        fit_m, tune_m = ([matrices[i] for i in indices] for indices in (fit_ids, tune_ids))
        models, info = core.fit_models(fit_rows, fit_m, core.PRIMARY)
        training.append({'fold': fold, 'fit': info, 'held_out_rows': len(tune_rows)})
        scores = {c: core.model_scores(tune_m, model, core.PRIMARY) for c, model in models.items()}
        for j, config in enumerate(grid):
            if config['C'] is not None and config['C'] not in models:
                rejected.add(j)
                continue
            choices = core.choose(tune_rows, tune_m, scores.get(config['C'], [None]*len(tune_rows)),
                                  config, core.PRIMARY)
            outcomes = outcome_rows(tune_rows, choices, 10, [m is not None for m in tune_m])
            summary = summarize(outcomes)
            counts[j].update({key: value for key, value in summary.items()
                              if isinstance(value, int) and not isinstance(value, bool)})
        print(f'CV fold {fold+1}/{folds} complete: {len(fit_rows)} fit, {len(tune_rows)} held out', flush=True)
    reports = []
    for j, (config, count) in enumerate(zip(grid, counts)):
        if j in rejected:
            reports.append({'config': config, 'status': 'rejected_missing_or_unconverged_fold_model'})
            continue
        require(count['rows'] == len(rows), 'CV must score each validation row exactly once')
        summary = dict(count)
        summary.update(accuracy=count['correct']/count['rows'],
                       baseline_accuracy=count['baseline_correct']/count['rows'],
                       net_accuracy_delta=count['net_correct']/count['rows'])
        reports.append({'config': config, 'status': 'complete', 'summary': summary})
    viable = [r for r in reports if r['status'] == 'complete']
    selected = max(viable, key=lambda r: tuning_key(r['summary']))
    return selected, reports, training


def fit(args):
    root = args.experiment
    protocol = experiment(root)
    out = root / 'model'
    require(not out.exists(), 'A new model already exists; do not overwrite or refit after test')
    code = implementation()
    runtime = core.versions()
    rows, hashes = load_complete(root, 'val')  # No test candidate/target file is opened.
    require(len(rows) == protocol['training_rows'], 'Entire official validation is required')
    print(f'Computing descriptors for all {len(rows)} validation rows', flush=True)
    matrices, support = core.prepare(rows)
    selected, grid, fold_training = cross_validate(rows, matrices, protocol['cv_folds'])
    final_models, final_training = core.fit_models(rows, matrices, core.PRIMARY)
    # Fit a new estimator even if validation selects an inactive frequency policy.
    chosen_c = selected['config']['C'] if selected['config']['C'] is not None else .1
    require(chosen_c in final_models, 'Selected final refit did not converge; test must remain unscored')
    config = selected['config']
    frozen = {
        'version': VERSION, 'model': final_models[chosen_c], 'config': config,
        'fitted_C': chosen_c, 'features': core.TARGETED_NAMES,
        'protocol_sha256': sha256(root / 'protocol.json'),
        'implementation_sha256': code, 'runtime_versions': runtime,
        'training_records_sha256': hashes, 'training_rows': len(rows),
        'frozen_at': time.time(), 'initialization': 'new estimator; no pilot weights loaded',
    }
    require(implementation() == code and core.versions() == runtime, 'Code/runtime changed during fitting')
    for path, digest in hashes.items():
        require(sha256(root / path) == digest, 'Training candidates changed during fitting')
    out.mkdir()
    joblib.dump(frozen, out / 'frozen.joblib')
    receipt = {key: value for key, value in frozen.items() if key != 'model'}
    receipt.update(model_sha256=sha256(out / 'frozen.joblib'),
                   training=final_training, cv_selected=selected,
                   cv_training=fold_training, support=dict(Counter(s['reason'] for s in support)),
                   weights=core.weights_report({'model': frozen['model']}, core.TARGETED_NAMES),
                   test_used_for_training_or_selection=False)
    write_json(out / 'frozen.json', receipt)
    write_json(out / 'cross-validation.json', {'selection': selected, 'grid': grid, 'folds': fold_training})
    for name in ('frozen.joblib', 'frozen.json', 'cross-validation.json'):
        (out / name).chmod(0o444)
    print(json.dumps({'model': str(out / 'frozen.joblib'), 'config': config,
                      'training': final_training, 'test_used': False}, indent=2), flush=True)


def evaluate(args):
    root = args.experiment
    protocol = experiment(root)
    model_path = root / 'model' / 'frozen.joblib'
    receipt = json.loads((root / 'model' / 'frozen.json').read_text())
    require(sha256(model_path) == receipt['model_sha256'], 'New model hash changed')
    frozen = joblib.load(model_path)
    require(frozen['version'] == VERSION and frozen['features'] == core.TARGETED_NAMES, 'Model schema differs')
    require(frozen['implementation_sha256'] == implementation() and frozen['runtime_versions'] == core.versions(),
            'Frozen implementation/runtime changed')
    require(frozen['protocol_sha256'] == sha256(root / 'protocol.json'), 'Frozen protocol changed')
    for key, value in frozen.items():
        if key != 'model':
            require(value == receipt[key], 'Model metadata/receipt mismatch')
    rows, hashes = load_complete(root, 'test')
    require(len(rows) == protocol['test_rows'], 'Entire frozen test sample is required')
    binding = {'model_sha256': receipt['model_sha256'], 'test_records_sha256': hashes,
               'test_manifest_sha256': protocol['manifest_sha256']['test-rows.json']}
    out = root / 'assessment'
    if out.exists():
        artifacts = json.loads((out / 'artifact-receipt.json').read_text())
        require(all(sha256(out / name) == digest for name, digest in artifacts.items()), 'Saved assessment changed')
        report = json.loads((out / 'comparison.json').read_text())
        require(report['binding'] == binding, 'Already assessed with different inputs')
        print(json.dumps(report, indent=2), flush=True)
        return report
    matrices, support = core.prepare(rows)
    scores = core.model_scores(matrices, frozen['model'], core.PRIMARY)
    choices = core.choose(rows, matrices, scores, frozen['config'], core.PRIMARY)
    covered = [m is not None for m in matrices]
    outcomes = outcome_rows(rows, choices, 10, covered)
    baseline = summarize(outcome_rows(rows, core.baseline(rows), 10, covered))
    summary = summarize(outcomes)
    summary['paired_group_bootstrap'] = cluster_bootstrap(outcomes)
    summary['net_accuracy_change_pp'] = 100 * summary['net_accuracy_delta']
    report = {
        'version': VERSION, 'binding': binding, 'rows': len(rows),
        'groups': len({r['split_group'] for r in rows}),
        'frequency': baseline, 'new_reranker': summary, 'selected_config': frozen['config'],
        'support': dict(Counter(s['reason'] for s in support)), 'evaluated_at': time.time(),
        'interpretation': 'Random 20k sample of uninspected, validation-disjoint test components; not the full official test score.',
        'training': 'All official validation rows accounted for; pairwise updates use rows with eligible comparisons.',
    }
    out.mkdir()
    with (out / 'decisions.jsonl').open('x') as stream:
        for row, outcome, status in zip(rows, outcomes, support):
            stream.write(json.dumps(dict(outcome, source_row_id=row['source_row_id'],
                                         source_split='test', support=status)) + '\n')
    for path, digest in hashes.items():
        require(sha256(root / path) == digest, 'Test candidates changed during evaluation')
    require(sha256(model_path) == receipt['model_sha256'], 'Model changed during evaluation')
    write_json(out / 'comparison.json', report)
    write_json(out / 'artifact-receipt.json', {
        name: sha256(out / name) for name in ('comparison.json', 'decisions.jsonl')})
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit', 'evaluate'))
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    (fit if args.action == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
