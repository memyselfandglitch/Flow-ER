#!/usr/bin/env python3
"""Freeze full-validation training and a random, uninspected 20k test sample.

Reuse verified validation candidates only. Never import old model weights or
choose test cases using targets, correctness, sample counts, or candidate ranks.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_full_validation import sha
from run_targeted_inference import extract_row, PREDICTION_NAME

SEED = 'flower-full-validation-test20k-20261029-v1'


def select_test(rows, size=20000):
    eligible = [r for r in rows if r['fresh_group_disjoint']]
    if len(eligible) < size or size <= 0:
        raise ValueError('Insufficient eligible test rows')
    def key(row):
        return hashlib.sha256(f'{SEED}:test:{row["source_row_id"]}'.encode()).hexdigest()
    return sorted((dict(r) for r in sorted(eligible, key=key)[:size]),
                  key=lambda r: r['source_row_id'])


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def read_bound_manifest(root, name, protocol):
    path = root / name
    if sha(path) != protocol['manifest_sha256'][name]:
        raise ValueError(f'Manifest changed: {path}')
    return [json.loads(line) for line in path.open()]


def write_parts(source, rows, output, split, size=4096):
    """Read the source in order; emit only selected rows, retaining source IDs."""
    wanted = {r['source_row_id']: r for r in rows}
    if len(wanted) != len(rows):
        raise ValueError('Duplicate source identities')
    parts, chunk, lines = {}, [], []
    def flush():
        name = f'{split}{len(parts):03d}'
        entries = [dict(row, row_id=i) for i, row in enumerate(chunk)]
        inputs = output / 'inputs' / f'{name}.txt'
        metadata = output / f'{name}-rows.json'
        with inputs.open('xb') as stream:
            stream.writelines(lines)
        write_json(metadata, entries)
        parts[name] = {'source_split': split, 'rows': len(entries),
                       'inputs_sha256': sha(inputs), 'row_manifest_sha256': sha(metadata)}
        inputs.chmod(0o444); metadata.chmod(0o444)
        chunk.clear(); lines.clear()
    seen = set()
    with Path(source).open('rb') as stream:
        for index, raw in enumerate(stream):
            if index not in wanted:
                continue
            meta = wanted[index]
            if not raw.endswith(b'\n') or hashlib.sha256(raw).hexdigest() != meta['source_line_sha256']:
                raise ValueError(f'Source changed at {split}:{index}')
            chunk.append(meta); lines.append(raw); seen.add(index)
            if len(chunk) == size:
                flush()
    if chunk:
        flush()
    if seen != set(wanted):
        raise ValueError('Missing selected source rows')
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path(
        'results/reranking/full-validation-new-model-20261029'))
    parser.add_argument('--old-pilot', type=Path, default=Path(
        'results/reranking/targeted-pilot-20261029'))
    parser.add_argument('--out', type=Path, default=Path(
        'results/reranking/full-validation-test20k-20261029'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Experiment already prepared; do not resample test membership')
    base = json.loads((args.base / 'protocol.json').read_text())
    val = read_bound_manifest(args.base, 'validation-training-rows.jsonl', base)
    all_test = read_bound_manifest(args.base, 'test-evaluation-rows.jsonl', base)
    test = select_test(all_test)
    if {r['split_group'] for r in val} & {r['split_group'] for r in test}:
        raise ValueError('Validation/test group leakage')
    for split in ('val', 'test'):
        p = base['sources'][split]
        if sha(p['source']) != p['source_sha256']:
            raise ValueError(f'Official {split} dataset changed')
    checkpoint = base['generator_checkpoint']
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Pretrained FlowER changed')
    old_identity = json.loads((args.old_pilot / 'inference-identity.json').read_text())
    old_receipt = json.loads((args.old_pilot / 'operating-receipt.json').read_text())
    if (old_identity['checkpoint_sha256'] != checkpoint['sha256'] or
            old_identity['config']['sample_size'] != 64):
        raise ValueError('Cached generator or sampling identity differs')
    cached, reuse = {}, {}
    for name in ('tune', 'audit'):
        pred = args.old_pilot / 'predictions' / name / PREDICTION_NAME
        meta = args.old_pilot / f'{name}-rows.json'
        old_selection = json.loads((args.old_pilot / 'selection.json').read_text())
        if (sha(pred) != old_receipt['completion'][name]['prediction_sha256'] or
                sha(meta) != old_selection['parts'][name]['row_manifest_sha256']):
            raise ValueError(f'Cached {name} candidates/metadata changed')
        original = json.loads(meta.read_text())
        raw = pred.read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('Partial cached prediction')
        predictions = raw.splitlines(keepends=True)
        if len(original) != len(predictions):
            raise ValueError('Cached prediction/metadata count mismatch')
        for row, prediction in zip(original, predictions):
            i = row['source_row_id']
            if row['source_split'] != 'val' or i in cached:
                raise ValueError('Non-validation or duplicate cached source row')
            for field in ('source_line_sha256', 'reactant_key', 'sequence_id'):
                if row[field] != val[i][field]:
                    raise ValueError('Cached input binding differs from official validation')
            cached[i] = prediction
        reuse[name] = {'prediction': str(pred), 'prediction_sha256': sha(pred),
                       'metadata': str(meta), 'metadata_sha256': sha(meta), 'rows': len(original)}
    if len(val) != 21723 or len(test) != 20000 or len(cached) != 1536:
        raise ValueError('Unexpected full experiment sizes')
    for row in val:
        row.update(row_id=row['source_row_id'],
                   candidate_origin='cached_validation' if row['source_row_id'] in cached else 'new_generation')
    for row in test:
        row.update(row_id=row['source_row_id'], candidate_origin='new_generation')
    args.out.mkdir(parents=True)
    (args.out / 'cache').mkdir()
    inference = args.out / 'inference'
    (inference / 'inputs').mkdir(parents=True)
    # Freeze cohort membership before extracting cached validation outcomes.
    for name, rows in (('validation-rows.json', val), ('test-rows.json', test)):
        write_json(args.out / name, rows)
        (args.out / name).chmod(0o444)
    protocol = {
        'schema': 1, 'status': 'frozen_before_new_inference_and_training', 'seed': SEED,
        'training_rows': len(val), 'test_rows': len(test), 'cv_folds': base['folds'],
        'validation_groups': len({r['split_group'] for r in val}),
        'test_groups': len({r['split_group'] for r in test}),
        'test_sampling_frame_rows': sum(r['fresh_group_disjoint'] for r in all_test),
        'test_selection': '20,000 smallest seeded SHA256 source-row hashes, then original source order',
        'test_exclusions': 'Components touching official validation or previously inspected test prefix; input metadata only',
        'inference_unit': 'supplied elementary step; not complete reaction pathway',
        'sample_size': 64, 'primary_k': 10,
        'generator_checkpoint': checkpoint,
        'new_reranker': 'fresh pairwise logistic estimator with 97 existing descriptors; no old learned weights loaded',
        'selection_policy': '5-fold grouped CV within validation; refit selected configuration on all validation',
        'negative_policy': 'skip alternate exact targets for same input observed in the current fitting partition only',
        'sources': base['sources'],
        'manifest_sha256': {n: sha(args.out / n) for n in ('validation-rows.json', 'test-rows.json')},
        'reuse': reuse, 'reused_validation_rows': len(cached), 'reused_test_rows': 0,
        'remaining_validation_rows': len(val)-len(cached), 'remaining_test_rows': len(test),
        'preparation_script_sha256': sha(__file__),
        'cached_inference_identity_sha256': sha(args.old_pilot / 'inference-identity.json'),
        'limitation': 'Prior feature design used diagnostic test cases; selected test components were not inspected or used for that design.',
    }
    write_json(args.out / 'protocol.json', protocol)
    (args.out / 'protocol.json').chmod(0o444)
    # Canonical extraction uses exactly the prior saved candidate strings/counts.
    cache_path = args.out / 'cache' / 'val.jsonl'
    with Path(base['sources']['val']['source']).open() as source, cache_path.open('x') as destination:
        for i, line in enumerate(source):
            if i in cached:
                row = extract_row(line, cached[i], i, val[i])
                destination.write(json.dumps(row, allow_nan=False) + '\n')
    cache_path.chmod(0o444)
    write_json(args.out / 'cache' / 'manifest.json', {
        'rows': len(cached), 'records_sha256': sha(cache_path),
        'source': 'verified prior validation raw predictions; no prior test candidates imported',
        'source_row_ids': sorted(cached), 'sample_size': 64,
    })
    parts = write_parts(base['sources']['val']['source'],
                        [r for r in val if r['source_row_id'] not in cached], inference, 'val')
    parts.update(write_parts(base['sources']['test']['source'], test, inference, 'test'))
    write_json(inference / 'selection.json', {
        'sample_size': 64, 'primary_k': 10, 'parts': parts,
        'protocol_sha256': sha(args.out / 'protocol.json'),
        'cached_validation_records_sha256': sha(cache_path),
        'generator_checkpoint': checkpoint,
    })
    (inference / 'selection.json').chmod(0o444)
    print(json.dumps({'experiment': str(args.out), 'reused_validation': len(cached),
                      'remaining_validation': len(val)-len(cached), 'test': len(test),
                      'sampling_frame': protocol['test_sampling_frame_rows'],
                      'inference_parts': list(parts)}, indent=2))


if __name__ == '__main__':
    main()
