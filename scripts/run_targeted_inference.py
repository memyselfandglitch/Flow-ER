#!/usr/bin/env python3
"""Run only explicitly selected, frozen pilot inputs; preserve raw candidate sets.

No dependency installation, energy scoring, model fitting or cloud lifecycle calls.
Resume is opt-in and rejects changed identities, foreign output, skipped inputs,
partial prediction lines or an unverified prediction prefix without truncating it.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from diagnose_rank_failures import analyze_saved, reference
from rerank_analysis import canonical

CHECKPOINT = Path('checkpoints/flower_new_dataset/best_large_hyperparam/model.2940000_97.pt')
PREDICTION_NAME = 'test-64-2940000_97.txt'
DEFAULT_PARTS = ('fit', 'tune', 'audit')
META_KEYS = ('source_row_id', 'source_split', 'sequence_id', 'raw_sequence_id',
             'split_group', 'reactant_key', 'source_line_sha256')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_parts(selection, argument):
    names = argument.split(',') if argument is not None else [
        name for name in DEFAULT_PARTS if name in selection['parts']]
    require(names and len(set(names)) == len(names), 'Select distinct pilot parts')
    for name in names:
        require(re.fullmatch(r'[a-z][a-z0-9_-]*', name) is not None,
                f'Unsafe part name: {name!r}')
        require(name in selection['parts'], f'Unknown pilot part: {name}')
    return names


def load_part(pilot, name, spec):
    inputs = pilot / 'inputs' / f'{name}.txt'
    metadata_path = pilot / f'{name}-rows.json'
    require(sha256(inputs) == spec['inputs_sha256'], f'{name}: input hash mismatch')
    require(sha256(metadata_path) == spec['row_manifest_sha256'],
            f'{name}: row manifest hash mismatch')
    raw = inputs.read_bytes()
    require(not raw or raw.endswith(b'\n'), f'{name}: incomplete input line')
    truth = raw.decode().splitlines(keepends=True)
    metadata = json.loads(metadata_path.read_text())
    require(isinstance(metadata, list) and len(truth) == len(metadata) == spec['rows'],
            f'{name}: input/metadata row counts differ')
    identities = []
    for index, (line, meta) in enumerate(zip(truth, metadata)):
        require(all(key in meta for key in META_KEYS if key != 'raw_sequence_id'),
                f'{name}:{index}: missing source metadata')
        require(meta['source_split'] == spec['source_split'], f'{name}: source split mismatch')
        require(digest_bytes(line.encode()) == meta['source_line_sha256'],
                f'{name}:{index}: source line hash mismatch')
        raw_input = line.split('|')[0].split('>>')[0]
        require(digest_bytes(canonical(raw_input).encode()) == meta['reactant_key'],
                f'{name}:{index}: canonical input hash mismatch')
        identities.append((meta['source_split'], meta['source_row_id']))
    require(len(set(identities)) == len(identities), f'{name}: duplicate source rows')
    return truth, metadata


def raw_prediction_lines(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    require(not raw or raw.endswith(b'\n'),
            f'{path}: incomplete final prediction line; preserved, not truncated')
    return raw.splitlines(keepends=True)


def extract_row(truth, prediction, index, meta):
    """Keep opaque candidates and writer order, while restoring source provenance."""
    # Validate the source's canonical key separately: the writer performs an
    # additional SMILES roundtrip which need not have the same serialized key.
    require(digest_bytes(truth.encode()) == meta['source_line_sha256'],
            f'Input alignment hash mismatch at pilot row {index}')
    raw_input = truth.split('|')[0].split('>>')[0]
    require(digest_bytes(canonical(raw_input).encode()) == meta['reactant_key'],
            f'Input identity mismatch at pilot row {index}')
    writer_key = digest_bytes(reference(raw_input).encode())
    adapted = dict(meta, reactant_key=writer_key)
    try:
        row = analyze_saved(truth, prediction.decode(), index, adapted, k=10)
    except (AssertionError, ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f'Invalid/alignment-mismatched prediction at pilot row {index}: {error}') from error
    # analyze_saved historically sets test/local source indices. Restore the
    # manifest's actual train/validation/test identity, never outcome fields.
    row.update({key: meta[key] for key in META_KEYS if key in meta})
    row.update(row_id=index, pilot_row_id=index, writer_reactant_key=writer_key,
               selection_metadata=meta)
    fields = prediction.decode().rstrip('\n').split('|', 2)
    row['saved_asymmetric_matrix_count'] = ast.literal_eval(fields[1])
    row['prediction_line_sha256'] = digest_bytes(prediction)
    return row


def verified_prefix(prediction, truth, metadata):
    lines = raw_prediction_lines(prediction)
    require(len(lines) <= len(truth), 'More predictions than frozen input rows')
    rows = [extract_row(truth[i], line, i, metadata[i]) for i, line in enumerate(lines)]
    return lines, rows


def verify_log(log_path, start, produced):
    text = log_path.read_text(errors='replace') if log_path.exists() else ''
    skipped = re.findall(r'([\d.]+)% data is unparseable', text)
    require(not any(float(value) != 0 for value in skipped),
            f'{log_path}: parser skipped input rows; alignment is unsafe')
    if produced:
        require(skipped, f'{log_path}: no zero-skip evidence for written predictions')
        offsets = re.findall(r'Continuing previous runs at reaction (\d+)\.\.\.', text)
        starts = text.count('Starting new run...')
        if start:
            require(offsets == [str(start)] and starts == 0,
                    f'{log_path}: resume offset does not match verified prefix')
        else:
            require((offsets == ['0'] and starts == 0) or (offsets == [] and starts == 1),
                    f'{log_path}: missing/ambiguous zero-offset start evidence')


def process_group_alive(pid):
    if not pid:
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def verify_attempts(pilot, name, lines, identity_hash):
    attempts = sorted((pilot / 'attempts').glob(f'{name}-*.json'))
    require(attempts or not lines, f'{name}: foreign predictions without attempt receipts')
    consumed = 0
    for index, path in enumerate(attempts):
        attempt = json.loads(path.read_text())
        require(attempt['identity_sha256'] == identity_hash, f'{name}: attempt identity changed')
        require(attempt['start_rows'] == consumed, f'{name}: noncontiguous attempt offsets')
        require(attempt['prefix_sha256'] == digest_bytes(b''.join(lines[:consumed])),
                f'{name}: saved prefix changed before attempt {index + 1}')
        if attempt['status'] == 'running':
            require(index == len(attempts) - 1, f'{name}: unfinished nonfinal attempt')
            require(not process_group_alive(attempt.get('pid')),
                    f'{name}: previous inference process group may still be active')
        end = attempt.get('end_rows', len(lines))
        require(consumed <= end <= len(lines), f'{name}: invalid attempt output range')
        if 'prediction_sha256' in attempt:
            require(attempt['prediction_sha256'] == digest_bytes(b''.join(lines[:end])),
                    f'{name}: attempt output prefix changed')
        verify_log(pilot / attempt['log'], consumed, end - consumed)
        consumed = end
    require(consumed == len(lines), f'{name}: predictions extend beyond verified attempts')
    return attempts


def stop_process(process, grace=10):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    # A launcher may exit before its workers. End any surviving group members.
    if process_group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_process(command, cwd, env, log_path, deadline, on_start, on_progress):
    require(deadline > time.monotonic(), 'Inference time budget already exhausted')
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            on_start(process.pid)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Targeted inference hard time limit reached')
                try:
                    code = process.wait(timeout=min(5.0, remaining))
                    on_progress()
                    require(code == 0, f'Inference exited {code}; inspect {log_path}')
                    return
                except subprocess.TimeoutExpired:
                    on_progress()
        finally:
            if process.poll() is None or process_group_alive(process.pid):
                stop_process(process)


def execute_part(root, pilot, name, spec, identity_hash, config, deadline,
                 state, save, launch=run_process):
    truth, metadata = load_part(pilot, name, spec)
    prediction = pilot / 'predictions' / name / PREDICTION_NAME
    prediction.parent.mkdir(parents=True, exist_ok=True)
    record_path = pilot / 'records' / f'{name}.jsonl'
    receipt_path = pilot / 'records' / f'{name}.manifest.json'
    lines, rows = verified_prefix(prediction, truth, metadata)
    attempts = verify_attempts(pilot, name, lines, identity_hash)
    # A killed runner can leave a running receipt after its GPU process group
    # has exited. Seal its verified prefix before creating a later attempt.
    if attempts:
        previous = json.loads(attempts[-1].read_text())
        if previous['status'] == 'running':
            previous.update(status='interrupted_prefix_verified', end_rows=len(lines),
                            prediction_sha256=digest_bytes(b''.join(lines)),
                            recovered_at=time.time())
            atomic_json(attempts[-1], previous)
    if record_path.exists() or receipt_path.exists():
        require(record_path.exists() and receipt_path.exists(),
                f'{name}: incomplete extraction receipt; preserve and inspect it')
        receipt = json.loads(receipt_path.read_text())
        require(receipt['identity_sha256'] == identity_hash and receipt['rows'] == len(truth)
                and len(lines) == len(truth), f'{name}: invalid completed extraction')
        require(sha256(record_path) == receipt['records_sha256'], f'{name}: extracted records changed')
        require(digest_bytes(b''.join(lines)) == receipt['prediction_sha256'],
                f'{name}: completed predictions changed')
        require(all(sha256(pilot / path) == expected
                    for path, expected in receipt['logs_sha256'].items()),
                f'{name}: completed inference logs changed')
        return receipt
    if len(lines) < len(truth):
        attempt_number = len(attempts) + 1
        attempt_path = pilot / 'attempts' / f'{name}-{attempt_number:04}.json'
        log_path = pilot / 'logs' / f'inference-{name}-attempt-{attempt_number:04}.log'
        attempt = {'status': 'running', 'identity_sha256': identity_hash,
                   'start_rows': len(lines), 'prefix_sha256': digest_bytes(b''.join(lines)),
                   'log': str(log_path.relative_to(pilot)), 'started_at': time.time()}
        atomic_json(attempt_path, attempt)
        state.update(phase=name, phase_started_at=time.time(), phase_log=attempt['log'])
        def started(pid):
            attempt['pid'] = pid
            atomic_json(attempt_path, attempt)
            state['phase_pid'] = pid
            save()
        def progress():
            raw = prediction.read_bytes() if prediction.exists() else b''
            state.setdefault('parts', {}).setdefault(name, {}).update(
                status='running', complete_prediction_lines=raw.count(b'\n'),
                expected_rows=len(truth), prediction_bytes=len(raw))
            save()
        env = dict(os.environ, **config['environment'],
                   TEST_FILE=str(pilot / 'inputs' / f'{name}.txt'),
                   RESULT_PATH=str(prediction.parent))
        # Use the current interpreter environment's torchrun, not a possibly
        # different Python environment inherited through PATH.
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        error = None
        try:
            launch(['bash', 'scripts/test_vast.sh'], root, env, log_path, deadline, started, progress)
        except BaseException as exc:
            error = exc
        try:
            lines, rows = verified_prefix(prediction, truth, metadata)
            require(digest_bytes(b''.join(lines[:attempt['start_rows']])) == attempt['prefix_sha256'],
                    f'{name}: verified prefix changed during inference')
            verify_log(log_path, attempt['start_rows'], len(lines) - attempt['start_rows'])
            attempt.update(end_rows=len(lines), prediction_sha256=digest_bytes(b''.join(lines)))
        except Exception as verification_error:
            attempt['verification_error'] = str(verification_error)
            if error is None:
                error = verification_error
        attempt.update(status='failed_or_incomplete' if error else 'completed', finished_at=time.time())
        if error:
            attempt['error'] = f'{type(error).__name__}: {error}'
        atomic_json(attempt_path, attempt)
        state.pop('phase_pid', None)
        save()
        if error:
            raise error
        require(len(lines) == len(truth), f'{name}: inference exited with an incomplete prefix')
    # Recheck frozen source files before extraction. Never append to records.
    load_part(pilot, name, spec)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = record_path.with_suffix('.jsonl.tmp')
    require(not temporary.exists(), f'{name}: existing extraction temporary file; inspect it')
    with temporary.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    temporary.replace(record_path)
    receipt = {'status': 'complete', 'identity_sha256': identity_hash, 'part': name,
               'rows': len(rows), 'sample_size': 64, 'primary_k': 10,
               'prediction_sha256': digest_bytes(b''.join(lines)),
               'records_sha256': sha256(record_path), 'input_sha256': spec['inputs_sha256'],
               'row_manifest_sha256': spec['row_manifest_sha256'],
               'opaque_candidate_rows': sum(not row['rerank_eligible'] for row in rows),
               'empty_candidate_rows': sum(not row['candidates'] for row in rows),
               'alignment': 'one GPU, unsorted sequential loader, zero parser skips per writing attempt, verified prefix and target counts',
               'logs_sha256': {str(path.relative_to(pilot)): sha256(path)
                               for path in sorted((pilot / 'logs').glob(f'inference-{name}-attempt-*.log'))}}
    atomic_json(receipt_path, receipt)
    return receipt


def configuration(args):
    require(args.batch_size > 0 and args.max_matrix_elements > 0 and args.workers > 0,
            'Batch size, matrix element limit and worker count must be positive')
    require(1 <= args.sample_chunk_size <= 64, 'Sample chunk size must be 1..64')
    return {'environment': {
        'NUM_GPUS_PER_NODE': '1', 'NUM_NODES': '1', 'SAMPLE_SIZE': '64',
        'TEST_BATCH_SIZE': str(args.batch_size), 'MAX_SAMPLE_MATRIX_ELEMENTS': str(args.max_matrix_elements),
        'SAMPLE_CHUNK_SIZE': str(args.sample_chunk_size), 'NUM_WORKERS': str(args.workers),
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': str(args.seed),
        'SEED': str(args.seed), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'ACCUMULATION_COUNT': '1', 'TRAIN_BATCH_SIZE': '4096', 'VAL_BATCH_SIZE': '4096',
    }, 'sample_size': 64, 'primary_k': 10,
       'resume_randomness': 'A resumed process may use a different stochastic stream; completed rows are retained exactly. No bitwise equivalence to uninterrupted generation is claimed.'}


def collect_identity(root, pilot, config, preserved, preserved_sha):
    code = [root / name for name in ('eval_multiGPU.py', 'train.py', 'settings.py',
            'scripts/test_vast.sh', 'scripts/run_targeted_inference.py',
            'scripts/diagnose_rank_failures.py', 'scripts/rerank_analysis.py',
            'scripts/summarize_saved_ranks.py', 'scripts/prepare_rerank_pilot.py')]
    code += list((root / 'utils').rglob('*.py')) + list((root / 'model').rglob('*.py'))
    packages = {}
    for name in ('torch', 'torchdiffeq', 'rdkit', 'numpy', 'scipy', 'iteround'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Include nonsecret inherited runtime controls that can affect inference.
    inherited = {key: value for key, value in os.environ.items()
                 if key.startswith(('CUDA_', 'NVIDIA_', 'TORCH_', 'CUBLAS_', 'CUDNN_', 'NCCL_'))
                 or key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD')}
    return {'schema': 1, 'root': str(root), 'pilot': str(pilot),
            'selection_sha256': sha256(pilot / 'selection.json'), 'config': config,
            'code_sha256': {str(path.relative_to(root)): sha256(path) for path in sorted(set(code))},
            'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha256(root / CHECKPOINT),
            'preserved_prediction': str(preserved), 'preserved_prediction_sha256': preserved_sha,
            'python': sys.version, 'python_executable': str(Path(sys.executable).resolve()),
            'packages': packages, 'inherited_runtime_environment': inherited}


def freeze_or_verify_identity(pilot, identity, resume):
    path = pilot / 'inference-identity.json'
    if path.exists():
        require(resume, 'Pilot inference already initialized; use --resume after reviewing saved status')
        require(json.loads(path.read_text()) == identity,
                'Source/config/runtime identity changed; refusing to resume or reuse outputs')
    else:
        require(not resume, 'No inference identity exists to resume')
        for directory in ('predictions', 'records', 'attempts'):
            require(not (pilot / directory).exists() or not any((pilot / directory).iterdir()),
                    f'Existing {directory} without an inference identity; refusing foreign outputs')
        require(not list((pilot / 'logs').glob('inference-*-attempt-*.log'))
                and not (pilot / 'inference-status.json').exists(),
                'Existing inference logs/status without identity; refusing foreign outputs')
        with path.open('x') as stream:
            json.dump(identity, stream, indent=2)
            stream.write('\n')
        path.chmod(0o444)
    return sha256(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--parts', help='Comma-separated part names; default selects only fit,tune,audit present in manifest')
    parser.add_argument('--max-seconds', type=float, default=7200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-size', type=int, default=512, help='Evaluator tokens_sum budget')
    parser.add_argument('--max-matrix-elements', type=int, default=4_000_000)
    parser.add_argument('--sample-chunk-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preserved-prediction', type=Path)
    parser.add_argument('--preserved-prediction-sha256')
    args = parser.parse_args(argv)
    require(__debug__, 'Run without Python -O: extraction integrity checks require assertions')
    require(args.max_seconds > 0, '--max-seconds must be positive')
    root = args.root.resolve()
    pilot = (args.pilot if args.pilot.is_absolute() else root / args.pilot).resolve()
    require(pilot.is_dir() and pilot != root, 'Pilot must be an existing dedicated directory')
    selection = json.loads((pilot / 'selection.json').read_text())
    require(selection.get('sample_size') == 64 and selection.get('primary_k') == 10,
            'Frozen selection must specify sample_size=64 and primary_k=10')
    names = selected_parts(selection, args.parts)
    require(bool(args.preserved_prediction) == bool(args.preserved_prediction_sha256),
            'Supply both preserved prediction path and SHA256, or neither')
    if args.preserved_prediction:
        preserved = args.preserved_prediction
        preserved = (preserved if preserved.is_absolute() else root / preserved).resolve()
        expected_sha = args.preserved_prediction_sha256
    else:
        paused = json.loads((root / '.vast' / 'paused-run.json').read_text())
        preserved = (root / paused['prediction']).resolve()
        expected_sha = paused['prediction_sha256']
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha) is not None, 'Invalid preserved SHA256')
    require(not preserved.is_relative_to(pilot), 'Preserved benchmark must be outside pilot outputs')
    require(sha256(preserved) == expected_sha, 'Paused full-benchmark prediction hash changed')
    if selection.get('diagnostic_prediction_sha256'):
        require(selection['diagnostic_prediction_sha256'] == expected_sha,
                'Selection diagnostic prediction identity disagrees with preserved file')
    config = configuration(args)
    started = time.time()
    deadline = time.monotonic() + args.max_seconds
    with (pilot / '.inference.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another targeted inference runner holds the pilot lock') from error
        identity = collect_identity(root, pilot, config, preserved, expected_sha)
        identity_hash = freeze_or_verify_identity(pilot, identity, args.resume)
        for directory in ('logs', 'attempts', 'records'):
            (pilot / directory).mkdir(exist_ok=True)
        status_path = pilot / 'inference-status.json'
        old = json.loads(status_path.read_text()) if status_path.exists() else {}
        state = {'status': 'running', 'identity_sha256': identity_hash,
                 'invocation_started_at': started, 'max_seconds': args.max_seconds,
                 'requested_parts': names, 'parts': old.get('parts', {}),
                 'full_benchmark_status': 'paused', 'phase': 'preflight'}
        def save():
            state['updated_at'] = time.time()
            atomic_json(status_path, state)
            atomic_json(pilot / 'inference-progress.json', {
                key: state[key] for key in ('status', 'phase', 'updated_at', 'parts')})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        save()
        code = 0
        try:
            # Validate every requested partition before spending GPU time.
            for name in names:
                load_part(pilot, name, selection['parts'][name])
            for name in names:
                receipt = execute_part(root, pilot, name, selection['parts'][name], identity_hash,
                                       config, deadline, state, save)
                state['parts'][name] = dict(receipt, complete_prediction_lines=receipt['rows'])
                save()
                print(f'{name}: {receipt["rows"]} rows extracted with raw candidate rankings preserved', flush=True)
            require(collect_identity(root, pilot, config, preserved, expected_sha) == identity,
                    'Code/config/checkpoint identity changed during inference')
            state.update(status='completed', phase='complete')
        except BaseException as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 1
            state.update(status='failed_or_incomplete', error=f'{type(error).__name__}: {error}')
            print(state['error'], file=sys.stderr, flush=True)
        finally:
            try:
                unchanged = sha256(preserved) == expected_sha
            except OSError:
                unchanged = False
            state['full_benchmark_prefix_unchanged'] = unchanged
            if not unchanged:
                code = 1
                state.update(status='failed_or_incomplete', preserved_prediction_error='Benchmark hash no longer matches')
            state['invocation_finished_at'] = time.time()
            save()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        return code


if __name__ == '__main__':
    sys.exit(main())
