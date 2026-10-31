#!/usr/bin/env python3
"""Adopt existing validation inference and finish with backup; never launch training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from full_run_remote_guard import stop_own_instance


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def process(pid, started):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] == str(started) and fields[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def verify_generation(root, experiment):
    pilot = experiment / 'inference'
    selection = json.loads((pilot / 'selection.json').read_text())
    protocol = json.loads((experiment / 'protocol.json').read_text())
    status = json.loads((pilot / 'inference-status.json').read_text())
    names = [name for name, spec in selection['parts'].items() if spec['source_split'] == 'val']
    if status['status'] != 'completed' or status['requested_parts'] != names:
        raise ValueError('Validation inference did not complete')
    if not status['full_benchmark_prefix_unchanged']:
        raise ValueError('Preserved benchmark prefix changed')
    if (experiment / 'model').exists():
        raise ValueError('Unexpected new reranker model directory')
    if any(p.stat().st_size for p in (pilot / 'predictions').glob('test*/*.txt')):
        raise ValueError('Unexpected test generation')
    if sha(root / protocol['generator_checkpoint']['path']) != protocol['generator_checkpoint']['sha256']:
        raise ValueError('Pretrained checkpoint changed')
    for name, digest in protocol['manifest_sha256'].items():
        if sha(experiment / name) != digest:
            raise ValueError('Frozen cohort changed: ' + name)
    counts = {}
    records = [(experiment / 'cache/val.jsonl', protocol['reused_validation_rows'],
                selection['cached_validation_records_sha256'])]
    for name in names:
        spec = selection['parts'][name]
        receipt = json.loads((pilot / 'records' / f'{name}.manifest.json').read_text())
        prediction = pilot / 'predictions' / name / 'test-64-2940000_97.txt'
        raw = prediction.read_bytes()
        if not (receipt['status'] == 'complete' and receipt['rows'] == spec['rows']
                and receipt['sample_size'] == 64 and raw.endswith(b'\n')
                and raw.count(b'\n') == spec['rows']
                and hashlib.sha256(raw).hexdigest() == receipt['prediction_sha256']
                and sha(pilot / 'inputs' / f'{name}.txt') == spec['inputs_sha256'] == receipt['input_sha256']
                and sha(pilot / f'{name}-rows.json') == spec['row_manifest_sha256'] == receipt['row_manifest_sha256']):
            raise ValueError('Incomplete or changed validation shard: ' + name)
        for log, digest in receipt['logs_sha256'].items():
            if sha(pilot / log) != digest:
                raise ValueError('Inference log changed: ' + log)
        records.append((pilot / 'records' / f'{name}.jsonl', spec['rows'], receipt['records_sha256']))
        counts[name] = spec['rows']
    for path, expected, digest in records:
        if sha(path) != digest:
            raise ValueError('Candidate records changed: ' + str(path))
        count = 0
        with path.open() as stream:
            for line in stream:
                if json.loads(line)['sample_count'] != 64:
                    raise ValueError('Candidate row does not contain 64 samples')
                count += 1
        if count != expected:
            raise ValueError('Candidate row count differs')
    total = sum(counts.values()) + protocol['reused_validation_rows']
    if total != protocol['training_rows']:
        raise ValueError('Incomplete validation dataset')
    return {'validation_rows': total, 'generated_by_part': counts,
            'cached_validation_rows': protocol['reused_validation_rows'],
            'sample_count': 64, 'reranker_training_started': False,
            'test_generation_started': False, 'verified_at': time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    parser.add_argument('--supervisor-start', required=True)
    parser.add_argument('--inference-pid', type=int, required=True)
    parser.add_argument('--inference-start', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    experiment = args.root / args.experiment
    with (experiment / '.generation-only.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not process(args.supervisor_pid, args.supervisor_start):
            raise RuntimeError('Original suspended supervisor identity differs')
        fields = Path(f'/proc/{args.supervisor_pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{args.supervisor_pid}/cmdline').read_bytes().split(b'\0')
        if fields[0] != 'T' or b'scripts/resume_full_validation_experiment.py' not in command:
            raise RuntimeError('Original supervisor is not safely suspended')
        # Only the suspended parent is removed. Inference owns a separate session.
        os.kill(args.supervisor_pid, signal.SIGKILL)
        state = json.loads((experiment / 'run-status.json').read_text())
        state.update(status='running', phase='validation_candidates', training_held_by_user=True,
                     test_generation_held_by_user=True, continuation_policy='validation_generation_only',
                     generation_only_monitor_pid=os.getpid(), policy_changed_at=time.time())
        write(experiment / 'run-status.json', state)
        write(experiment / 'generation-only-armed.json', {
            'pid': os.getpid(), 'inference_pid': args.inference_pid,
            'original_supervisor_terminated': args.supervisor_pid, 'armed_at': time.time(),
            'deadline': args.deadline, 'training_enabled': False, 'test_generation_enabled': False})
        try:
            while process(args.inference_pid, args.inference_start):
                if time.time() >= args.deadline - 120:
                    raise TimeoutError('Validation generation exceeded the spending deadline')
                time.sleep(5)
            receipt = verify_generation(args.root, experiment)
            write(experiment / 'generation-only-complete.json', receipt)
            state.update(status='validation_generation_completed', phase='waiting_for_verified_backup',
                         validation_rows=receipt['validation_rows'])
        except Exception as error:
            state.update(status='generation_incomplete', phase='waiting_for_verified_backup',
                         error=type(error).__name__ + ': ' + str(error))
        state.update(finished_work_at=time.time())
        write(experiment / 'run-status.json', state)
        write(experiment / 'ready-for-backup.json', {'status': state['status'], 'at': time.time(),
                                                   'scope': 'validation_generation_only'})
        # Allow final copying while SSH is available; retain disk if the client is offline.
        until = min(time.time() + 1200, args.deadline - 60)
        while time.time() < until and not (experiment / 'backup-verified.json').exists():
            time.sleep(5)
        state.update(stop_requested_at=time.time(),
                     backup_acknowledged=(experiment / 'backup-verified.json').exists())
        write(experiment / 'run-status.json', state)
        for _ in range(12):
            try:
                stop_own_instance('/root/flower-instance-env.json')
                return
            except Exception as error:
                print(type(error).__name__, flush=True)
                time.sleep(5)


if __name__ == '__main__':
    main()
