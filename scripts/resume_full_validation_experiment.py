#!/usr/bin/env python3
"""Resume the frozen experiment while preserving all previous run artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'
import run_full_validation_experiment as supervisor
import run_full_inference as inference
import run_targeted_inference as base


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_no_inference_processes():
    names = {'eval_multiGPU.py', 'run_targeted_inference.py', 'run_full_inference.py',
             'run_full_validation_experiment.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            words = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not any(Path(word.decode(errors='replace')).name in names for word in words if word),
                'An existing inference/supervisor process must be inspected before resume')


def preserve_and_verify(root, experiment, history):
    pilot = experiment / 'inference'
    identity = json.loads((pilot / 'inference-identity.json').read_text())
    current = inference.identity(root, pilot, identity['config'],
                                 Path(identity['preserved_prediction']),
                                 identity['preserved_prediction_sha256'])
    require(current == identity, 'Frozen inference source, runtime, or configuration changed')
    identity_hash = base.sha256(pilot / 'inference-identity.json')
    selection = json.loads((pilot / 'selection.json').read_text())
    hashes = {}
    counts = {}
    for part, spec in selection['parts'].items():
        prediction = pilot / 'predictions' / part / base.PREDICTION_NAME
        truth, metadata = base.load_part(pilot, part, spec)
        lines, rows = base.verified_prefix(prediction, truth, metadata)
        require(all(row['sample_count'] == 64 for row in rows), 'Incomplete candidate samples')
        attempts = sorted((pilot / 'attempts').glob(f'{part}-*.json'))
        for path in attempts:
            previous = json.loads(path.read_text())
            if previous['status'] != 'running':
                continue
            require(path == attempts[-1], 'Only the final inference attempt may be interrupted')
            start = previous['start_rows']
            require(previous['identity_sha256'] == identity_hash and start <= len(lines),
                    'Interrupted attempt does not match this experiment')
            require(previous['prefix_sha256'] == base.digest_bytes(b''.join(lines[:start])),
                    'Interrupted attempt prefix changed')
            base.verify_log(pilot / previous['log'], start, len(lines) - start)
            shutil.copy2(path, history / path.name)
            previous.update(status='interrupted_prefix_verified', end_rows=len(lines),
                            prediction_sha256=base.digest_bytes(b''.join(lines)),
                            recovered_at=time.time(), recovery_history=str(history.relative_to(experiment)))
            base.atomic_json(path, previous)
        base.verify_attempts(pilot, part, lines, identity_hash)
        if prediction.exists():
            hashes[str(prediction.relative_to(experiment))] = base.sha256(prediction)
        counts[part] = len(lines)
    for name in ('run-status.json', 'ready-for-backup.json', 'backup-verified.json',
                 'supervisor-process.json'):
        path = experiment / name
        if path.exists():
            path.rename(history / name)
    receipt = {'verified_at': time.time(), 'rows': counts, 'prediction_sha256': hashes,
               'identity_sha256': identity_hash, 'predictions_modified': False}
    base.atomic_json(history / 'verified-prefixes.json', receipt)
    return receipt


def verify_existing_model(experiment):
    import full_validation_reranker as learner
    receipt = json.loads((experiment / 'model/frozen.json').read_text())
    require(receipt['model_sha256'] == base.sha256(experiment / 'model/frozen.joblib'),
            'Saved reranker weights changed')
    require(receipt['protocol_sha256'] == base.sha256(experiment / 'protocol.json'),
            'Saved reranker protocol changed')
    require(receipt['implementation_sha256'] == learner.implementation()
            and receipt['runtime_versions'] == learner.core.versions(),
            'Saved reranker implementation or runtime changed')
    for name, digest in receipt['training_records_sha256'].items():
        require(base.sha256(experiment / name) == digest, 'Saved reranker training candidates changed')


def resume_stages(experiment, suffix, runner):
    def run(command, log, deadline):
        command = list(command)
        if 'scripts/run_full_inference.py' in command and '--resume' not in command:
            command.append('--resume')
        if 'scripts/full_validation_reranker.py' in command and 'fit' in command \
                and (experiment / 'model').exists():
            verify_existing_model(experiment)
            return
        runner(command, log.with_name(f'{log.stem}-{suffix}{log.suffix}'), deadline)
        if 'scripts/run_full_inference.py' in command:
            status = json.loads((experiment / 'inference/inference-status.json').read_text())
            requested = command[command.index('--parts') + 1].split(',')
            require(status['status'] == 'completed' and status['requested_parts'] == requested,
                    'Inference stage did not finish; retaining its saved prefix')
    return run


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args, _ = parser.parse_known_args()
    root = args.root.resolve()
    experiment = (args.experiment if args.experiment.is_absolute() else root / args.experiment).resolve()
    with (experiment / '.resume-supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_no_inference_processes()
        suffix = f'resume-{time.time_ns()}'
        history = experiment / 'recovery' / suffix
        history.mkdir(parents=True)
        receipt = preserve_and_verify(root, experiment, history)
        print(json.dumps({'resume_verified': receipt, 'history': str(history)}), flush=True)
        supervisor.run_stage = resume_stages(experiment, suffix, supervisor.run_stage)
        return supervisor.main()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        try:
            supervisor.stop_own_instance('/root/flower-instance-env.json')
        except Exception:
            pass
        raise
