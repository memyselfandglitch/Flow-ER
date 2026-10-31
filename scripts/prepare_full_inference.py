#!/usr/bin/env python3
"""Materialize fresh-inference shards from the full validation/test protocol.

No previous predictions or reranker weights are read. Each original row appears
in exactly one shard. Sharding bounds runner memory; it does not select a subset.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_full_validation import sha


def write_shards(source, metadata, output, prefix, shard_rows):
    if shard_rows <= 0:
        raise ValueError('Shard size must be positive')
    parts, consumed = {}, 0
    with Path(source).open('rb') as stream:
        for start in range(0, len(metadata), shard_rows):
            name = f'{prefix}{start // shard_rows:03d}'
            entries = [dict(row, row_id=i) for i, row in
                       enumerate(metadata[start:start + shard_rows])]
            input_path = output / 'inputs' / f'{name}.txt'
            with input_path.open('xb') as destination:
                for entry in entries:
                    raw = stream.readline()
                    if (not raw.endswith(b'\n') or entry['source_row_id'] != consumed or
                            hashlib.sha256(raw).hexdigest() != entry['source_line_sha256']):
                        raise ValueError(f'{prefix}: incorrect source alignment at {consumed}')
                    destination.write(raw)
                    consumed += 1
            manifest_path = output / f'{name}-rows.json'
            manifest_path.write_text(json.dumps(entries) + '\n')
            parts[name] = {
                'source_split': entries[0]['source_split'], 'rows': len(entries),
                'source_row_start': start, 'source_row_stop_exclusive': consumed,
                'inputs_sha256': sha(input_path),
                'row_manifest_sha256': sha(manifest_path),
            }
            input_path.chmod(0o444)
            manifest_path.chmod(0o444)
        if stream.read(1) or consumed != len(metadata):
            raise ValueError(f'{prefix}: source length differs from complete manifest')
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path(
        'results/reranking/full-validation-new-model-20261029'))
    parser.add_argument('--shard-rows', type=int, default=4096)
    args = parser.parse_args()
    root = args.experiment
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    output = root / 'inference'
    if output.exists():
        raise FileExistsError('Inference preparation already exists; preserve its identity')
    checkpoint = protocol['generator_checkpoint']
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Pretrained FlowER checkpoint changed')
    metadata = {}
    for split, name in (('val', 'validation-training-rows.jsonl'),
                        ('test', 'test-evaluation-rows.jsonl')):
        path = root / name
        if sha(path) != protocol['manifest_sha256'][name]:
            raise ValueError(f'{split}: protocol manifest changed')
        source = protocol['sources'][split]
        if sha(source['source']) != source['source_sha256']:
            raise ValueError(f'{split}: source data changed')
        metadata[split] = [json.loads(line) for line in path.open()]
    (output / 'inputs').mkdir(parents=True)
    parts = {}
    for split in ('val', 'test'):
        parts.update(write_shards(protocol['sources'][split]['source'], metadata[split],
                                  output, split, args.shard_rows))
    selection = {
        'status': 'prepared_fresh_generation_not_started',
        'sample_size': protocol['sample_size'], 'primary_k': protocol['primary_k'],
        'protocol_sha256': sha(protocol_path),
        'generator_checkpoint': checkpoint,
        'source_sha256': {s: p['source_sha256'] for s, p in protocol['sources'].items()},
        'preparation_script_sha256': sha(__file__),
        'old_predictions_imported': 0, 'parts': parts,
    }
    path = output / 'selection.json'
    path.write_text(json.dumps(selection, indent=2) + '\n')
    path.chmod(0o444)
    print(json.dumps({
        'selection': str(path), 'rows': sum(p['rows'] for p in parts.values()),
        'validation_parts': [n for n in parts if n.startswith('val')],
        'test_parts': [n for n in parts if n.startswith('test')],
        'old_predictions_imported': 0,
    }, indent=2))


if __name__ == '__main__':
    main()
