#!/usr/bin/env python3
"""Detached val inference -> fresh CPU reranker -> test inference -> assessment.

The caller provisions the host, arms an independent deadline guard, and backs up
artifacts. This process survives SSH loss. It stops its own GPU after completion,
failure or deadline; deletion occurs only after a verified external backup.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'
from full_run_remote_guard import stop_own_instance
from run_targeted_inference import atomic_json, sha256


def run_stage(command, log, deadline):
    remaining = deadline-time.time()-180
    if remaining < 60:
        raise TimeoutError('Insufficient time before the independent hard deadline')
    with log.open('x') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=stream, start_new_session=True)
        try:
            result = process.wait(timeout=remaining)
            if result:
                raise RuntimeError(f'Stage failed with exit {result}: {log}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--preserved-prediction', type=Path, required=True)
    parser.add_argument('--preserved-sha256', required=True)
    args = parser.parse_args()
    os.chdir(args.root)
    root = args.experiment.resolve()
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    state_path = root / 'run-status.json'
    if state_path.exists():
        raise RuntimeError('Existing run status; inspect and preserve before any explicit resume')
    selection = json.loads((root / 'inference' / 'selection.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha256(args.root / protocol['generator_checkpoint']['path']) != protocol['generator_checkpoint']['sha256']:
        raise ValueError('Pretrained generator checkpoint changed')
    state = {'status': 'running', 'started_at': time.time(), 'deadline': args.deadline,
             'phase': 'validation_candidates', 'flow_er_frozen': True}
    atomic_json(state_path, state)
    completed = False
    try:
        for split in ('val', 'test'):
            if split == 'test':
                state.update(phase='fresh_reranker_fit'); atomic_json(state_path, state)
                run_stage([sys.executable, 'scripts/full_validation_reranker.py', 'fit',
                           '--experiment', str(root)], logs / 'fit.log', args.deadline)
                state.update(model_frozen_at=time.time(),
                             model_sha256=sha256(root / 'model' / 'frozen.joblib'))
            state.update(phase='validation_candidates' if split == 'val' else 'test_candidates')
            atomic_json(state_path, state)
            parts = [name for name, spec in selection['parts'].items() if spec['source_split'] == split]
            command = [sys.executable, 'scripts/run_full_inference.py',
                       '--root', str(args.root), '--pilot', str(root / 'inference'),
                       '--parts', ','.join(parts), '--max-seconds', str(int(args.deadline-time.time()-240)),
                       '--batch-size', '2048', '--max-matrix-elements', '10000000',
                       '--sample-chunk-size', '64', '--workers', '8',
                       '--preserved-prediction', str(args.preserved_prediction),
                       '--preserved-prediction-sha256', args.preserved_sha256]
            if split == 'test':
                command.append('--resume')
            run_stage(command, logs / f'{split}-inference.log', args.deadline)
        state.update(phase='test_assessment'); atomic_json(state_path, state)
        run_stage([sys.executable, 'scripts/full_validation_reranker.py', 'evaluate',
                   '--experiment', str(root)], logs / 'evaluate.log', args.deadline)
        completed = True
        state.update(status='completed', phase='waiting_for_verified_backup')
    except BaseException as exc:
        state.update(status='failed_or_incomplete', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        state.update(finished_work_at=time.time())
        atomic_json(state_path, state)
        atomic_json(root / 'ready-for-backup.json', {'status': state['status'], 'at': time.time()})
        # Keep SSH available briefly for the periodic external backup. If the
        # client is offline, stop compute and retain the disk for later recovery.
        backup_until = min(time.time()+1200, args.deadline-60)
        while time.time() < backup_until and not (root / 'backup-verified.json').exists():
            time.sleep(min(10, max(0, backup_until-time.time())))
        state.update(stop_requested_at=time.time(), backup_acknowledged=(root / 'backup-verified.json').exists())
        atomic_json(state_path, state)
        for _ in range(12):
            try:
                stop_own_instance('/root/flower-instance-env.json')
                break
            except Exception as exc:
                print('Self-stop retry:', type(exc).__name__, flush=True)
                time.sleep(5)
    return 0 if completed else 1


if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        # Startup/preflight failures occur before main's stage-level finally.
        # Do not leave an idle paid GPU running until the distant deadline.
        try:
            stop_own_instance('/root/flower-instance-env.json')
        except Exception:
            pass  # The independently armed local and remote guards remain.
        raise
    raise SystemExit(result)
