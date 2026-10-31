#!/usr/bin/env python3
"""Prepare all official validation/test identities without reading predictions.

This does not load a trained reranker, fit a model, or launch GPU inference.
Only input identities and source-local network IDs determine folds/cohorts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_rerank_pilot import joined_groups

SEED = 'flower-full-validation-new-model-20261029-v1'
FOLDS = 5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_metadata(validation, test, inspected_prefix):
    # Work on copies: previous pilot identities and results remain unchanged.
    validation, test = ([dict(row) for row in rows] for rows in (validation, test))
    for split, rows in (('val', validation), ('test', test)):
        if not rows or any(row['source_split'] != split or row['source_row_id'] != i
                           for i, row in enumerate(rows)):
            raise ValueError(f'{split}: expected every source row exactly once in order')
    if not 0 <= inspected_prefix <= len(test):
        raise ValueError('Invalid inspected test prefix')
    # No test labels enter grouping. Source-local network identifiers do not
    # imply that equally numbered networks in different splits are the same.
    joined_groups(validation + test)
    val_groups = {row['split_group'] for row in validation}
    val_inputs = {row['reactant_key'] for row in validation}
    diagnostic_groups = {row['split_group'] for row in test[:inspected_prefix]}
    for row in validation:
        value = hashlib.sha256((SEED + ':' + row['split_group']).encode()).hexdigest()
        row.update(cv_fold=int(value[:16], 16) % FOLDS, role='reranker_training')
    for row in test:
        inspected = row['source_row_id'] < inspected_prefix
        overlap = row['split_group'] in val_groups
        row.update(role='evaluation_only', previously_inspected=inspected,
                   input_seen_in_validation=row['reactant_key'] in val_inputs,
                   group_seen_in_validation=overlap,
                   group_touches_inspected_test=row['split_group'] in diagnostic_groups,
                   fresh_group_disjoint=(not inspected and not overlap and
                                         row['split_group'] not in diagnostic_groups))
    summary = {
        'validation_rows': len(validation), 'test_rows': len(test),
        'validation_groups': len(val_groups),
        'test_groups': len({row['split_group'] for row in test}),
        'validation_fold_rows': dict(sorted(Counter(row['cv_fold'] for row in validation).items())),
        'previously_inspected_test_rows': inspected_prefix,
        'test_exact_input_overlap_with_validation': sum(row['input_seen_in_validation'] for row in test),
        'test_group_overlap_with_validation': sum(row['group_seen_in_validation'] for row in test),
        'fresh_group_disjoint_test_rows': sum(row['fresh_group_disjoint'] for row in test),
        'excluded_validation_rows': 0, 'excluded_test_rows': 0,
    }
    return validation, test, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/flower_new_dataset'))
    parser.add_argument('--index-root', type=Path, default=Path('results/reranking/pilot-v2'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'checkpoints/flower_new_dataset/best_large_hyperparam/model.2940000_97.pt'))
    parser.add_argument('--out', type=Path,
                        default=Path('results/reranking/full-validation-new-model-20261029'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory; prepared manifests are preserved')
    receipt = json.loads((args.index_root / 'selection.json').read_text())
    checkpoint = {'path': str(args.checkpoint), 'sha256': sha(args.checkpoint),
                  'policy': 'existing pretrained FlowER; frozen, no replacement or fine-tuning'}
    metadata, provenance = {}, {}
    for split in ('val', 'test'):
        source = args.data / f'{split}.txt'
        index = args.index_root / f'{split}-input-index.jsonl'
        if sha(source) != receipt['source_sha256'][split]:
            raise ValueError(f'{split}: official source differs from indexed dataset')
        rows = [json.loads(line) for line in index.open()]
        if len(rows) != receipt['source_rows'][split]:
            raise ValueError(f'{split}: incomplete input index')
        # Check the input-only index's line bindings without parsing outcomes.
        with source.open('rb') as stream:
            count = 0
            for count, raw in enumerate(stream, 1):
                if count > len(rows) or hashlib.sha256(raw).hexdigest() != rows[count-1]['source_line_sha256']:
                    raise ValueError(f'{split}: source/index alignment differs at line {count}')
        if count != len(rows):
            raise ValueError(f'{split}: source/index lengths differ')
        metadata[split] = rows
        provenance[split] = {'source': str(source), 'source_sha256': sha(source),
                             'input_index': str(index), 'input_index_sha256': sha(index)}
    val, test, summary = prepare_metadata(metadata['val'], metadata['test'],
                                         receipt['diagnostic_prefix_rows'])
    args.out.mkdir(parents=True)
    hashes = {}
    for name, rows in (('validation-training-rows', val), ('test-evaluation-rows', test)):
        path = args.out / f'{name}.jsonl'
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        hashes[path.name] = sha(path)
        path.chmod(0o444)
    protocol = {
        'status': 'dataset_preparation_only_no_new_training_or_inference',
        'seed': SEED, 'folds': FOLDS, 'summary': summary,
        'sources': provenance, 'manifest_sha256': hashes,
        'preparation_script_sha256': sha(__file__),
        'new_reranker_initialization': 'from scratch; no pilot model weights loaded',
        'candidate_generation_scope': 'fresh 64-sample candidate sets; no old candidate predictions imported',
        'generator_checkpoint': checkpoint,
        'sample_size': 64, 'primary_k': 10,
        'architecture': 'not selected by this dataset-preparation script',
        'model_selection': 'grouped cross-validation within official validation only',
        'final_fit': 'refit selected configuration on all official validation rows',
        'evaluation': 'frequency versus frozen reranker on every official test row; count rescues and harms',
        'grouping': 'transitive same canonical input or source-local network over all val/test input metadata',
        'test_reporting': 'full official score plus predeclared inspection/validation-overlap strata',
        'limitations': [
            'Previously inspected test examples influenced prior feature design; regenerated samples do not erase that exposure.',
            'Official validation/test contain repeated inputs; full test is not wholly independent of reranker training.',
            'Rows with absent targets or no eligible negative still count toward all-row reporting but cannot supply a positive-negative training comparison.',
        ],
    }
    path = args.out / 'protocol.json'
    path.write_text(json.dumps(protocol, indent=2) + '\n')
    path.chmod(0o444)
    print(json.dumps({'output': str(args.out), **summary}, indent=2))


if __name__ == '__main__':
    main()
