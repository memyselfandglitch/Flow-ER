#!/usr/bin/env python3
"""Bounded GFN2-xTB pilot: additive isolated-fragment electronic energies.

This is an exploratory physical heuristic, not a barrier or free-energy model.
All candidates use the same geometry protocol. Radicals abstain (spin unknown).
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDetermineBonds
from rerank_analysis import PARSER, canonical, load_records


def inventory(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, PARSER))
    return sorted(Counter((a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms()).items()), Chem.GetFormalCharge(mol)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def computation_protocol(args):
    return {'protocol': 'ETKDGv3_xtb_opt_normal_closed_shell_connectivity_checked_v2',
            'method': 'GFN2-xTB', 'solvent': args.solvent, 'conformers': args.conformers,
            'seed': 42, 'threads_per_fragment': 1, 'optimization': 'normal', 'unpaired_electrons': 0,
            'timeout_seconds': args.timeout, 'timeout_scope': 'entire fragment child process including embedding and xTB',
            'max_heavy_atoms': args.max_heavy_atoms, 'xtb_version': args.xtb_version,
            'xtb_binary_sha256': args.xtb_sha256, 'rdkit_version': rdBase.rdkitVersion,
            'scorer_sha256': file_hash(__file__),
            'canonicalization_sha256': file_hash(Path(__file__).with_name('rerank_analysis.py')),
            'energy_units': 'hartree', 'aggregation': 'sum of independently optimized isolated-fragment energies',
            'spin_policy': 'closed shell only; abstain for explicit radicals or odd electron count',
            'isotope_policy': 'abstain because XYZ protocol does not preserve isotope information',
            'connectivity_policy': 'RDKit DetermineBonds followed by canonical explicit-H graph equality'}


def fragment_configuration(smiles, args):
    return {'smiles': smiles, 'method': 'GFN2-xTB', 'solvent': args.solvent,
            'computation_protocol': computation_protocol(args)}


def cache_path(configuration, args):
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    return Path(args.cache)/(digest+'.json')


def atomic_json(path, value):
    """Concurrent readers see either the previous complete cache record or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+path.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def failure(configuration, reason, exception=None):
    result = {**configuration, 'status': 'failed', 'energy_hartree': None, 'reason': reason}
    if exception is not None:
        result['exception_type'] = type(exception).__name__
    return result


def _calculate_fragment(smiles, args):
    """Run only inside the killable fragment child process."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, PARSER))
    result = {'status': 'unsupported', 'energy_hartree': None}
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        result['reason']='unassigned_spin_radical'
    elif (sum(a.GetAtomicNum() for a in mol.GetAtoms())-Chem.GetFormalCharge(mol)) % 2:
        result['reason']='odd_electron_count_requires_unassigned_spin'
    elif any(a.GetIsotope() for a in mol.GetAtoms()):
        result['reason']='isotope_not_supported_by_this_xyz_protocol'
    elif mol.GetNumHeavyAtoms() > args.max_heavy_atoms:
        result['reason']='size_limit'
    else:
        charge = Chem.GetFormalCharge(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1
        conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=args.conformers, params=params))
        energies, failures = [], []
        for conf in conformers:
            with tempfile.TemporaryDirectory(prefix='flower-xtb-') as work:
                geometry = Path(work)/'input.xyz'
                Chem.MolToXYZFile(mol,str(geometry),confId=conf)
                cmd = [args.xtb,str(geometry),'--gfn','2','--chrg',str(charge),'--uhf','0','--opt','normal']
                if args.solvent != 'gas':cmd.extend(['--alpb',args.solvent])
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                try:
                    run = subprocess.run(cmd,cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    failures.append('timeout');continue
                found = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)',run.stdout)
                optimized = Path(work)/'xtbopt.xyz'
                if run.returncode or not found or not optimized.exists() or 'GEOMETRY OPTIMIZATION CONVERGED' not in run.stdout.upper():
                    failures.append('xtb_not_converged');continue
                try:
                    inferred = Chem.MolFromXYZFile(str(optimized))
                    rdDetermineBonds.DetermineBonds(inferred,charge=charge)
                    if canonical(Chem.MolToSmiles(inferred)) != canonical(smiles):
                        failures.append('connectivity_changed_or_perception_failed');continue
                except Exception:
                    failures.append('connectivity_perception_failed');continue
                energy = float(found[-1])
                if not math.isfinite(energy):
                    failures.append('nonfinite_energy'); continue
                energies.append(energy)
        result.update(status='ok' if energies else 'failed', energy_hartree=min(energies) if energies else None,
                      converged_conformers=len(energies), attempted_conformers=len(conformers), failures=failures)
    return result


def _compute_bounded(smiles, args):
    # A separate session lets the wall timeout stop both the Python geometry
    # worker and all xTB descendants. Future.cancel() alone cannot do that.
    settings = {name: getattr(args, name) for name in ('xtb', 'solvent', 'conformers', 'timeout', 'max_heavy_atoms')}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_fragment-worker'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, _ = process.communicate(json.dumps({'smiles': smiles, 'settings': settings}), timeout=args.timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        return {'status': 'failed', 'energy_hartree': None, 'reason': 'fragment_wall_timeout'}
    if process.returncode:
        return {'status': 'failed', 'energy_hartree': None, 'reason': 'fragment_worker_failed',
                'worker_returncode': process.returncode}
    return json.loads(stdout)


def fragment_energy(smiles, args):
    configuration = fragment_configuration(smiles, args)
    path = cache_path(configuration, args)
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if all(cached.get(key) == value for key, value in configuration.items()) and valid_result(cached):
                return cached
        except (OSError, ValueError, TypeError):
            pass  # A partial/corrupt old record is never used as a scientific result.
    try:
        result = {**_compute_bounded(smiles, args), **configuration}
        if not valid_result(result):
            raise ValueError('Invalid fragment result schema')
    except Exception as exc:
        result = failure(configuration, 'fragment_exception', exc)
    try:
        atomic_json(path, result)
    except Exception as exc:
        result['cache_write_exception_type'] = type(exc).__name__
    return result


def valid_result(result):
    if not isinstance(result, dict) or result.get('status') not in {'ok', 'unsupported', 'failed'}:
        return False
    if result['status'] == 'ok':
        try:
            value = result['energy_hartree']
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        except (KeyError, ValueError, TypeError):
            return False
    return result.get('energy_hartree') is None


def select_records(records, max_rows):
    if max_rows < 0:
        raise ValueError('--max-rows must be zero (all rows) or a positive count')
    identifiers = [str(row['row_id']) for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('row_id must be unique within an input cohort')
    if not max_rows or max_rows >= len(records):
        return list(records)
    # Membership depends only on row ID. Retain input order for all output rows.
    selected = set(sorted(identifiers, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:max_rows])
    return [row for row in records if str(row['row_id']) in selected]


def prepare_candidates(records, args):
    planned, unique_fragments = [], set()
    for row in records:
        seen = set()
        for candidate in row['candidates'][:args.k]:
            smi = candidate['smiles']
            result = {'row_id': row['row_id'], 'smiles': smi, 'method': 'GFN2-xTB',
                      'solvent': args.solvent, 'status': 'unsupported', 'energy_hartree': None,
                      'interpretation': 'sum of optimized isolated-fragment electronic energies; not Gibbs free energy or a barrier'}
            fragments = None
            try:
                signature = canonical(smi)
                if signature in seen:
                    raise ValueError('Equivalent candidates must be merged before physical scoring')
                seen.add(signature)
                if inventory(smi) != inventory(row['reactant']):
                    result['reason'] = 'element_isotope_or_net_charge_mismatch'
                else:
                    mol = Chem.MolFromSmiles(smi, PARSER)
                    fragments = [canonical(Chem.MolToSmiles(fragment))
                                 for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)]
                    if not fragments:
                        raise ValueError('Candidate has no fragments')
                    unique_fragments.update(fragments)
            except Exception as exc:
                result.update(status='failed', reason='candidate_preparation_exception', exception_type=type(exc).__name__)
            planned.append((result, fragments))
    return planned, sorted(unique_fragments)


def run_unique_fragments(fragments, args, scorer=None, executor_factory=ProcessPoolExecutor, progress=print):
    scorer = scorer or fragment_energy
    results, counts = {}, Counter()
    total = len(fragments)

    def record(smi, value):
        if not valid_result(value):
            value = failure(fragment_configuration(smi, args), 'invalid_fragment_result')
        results[smi] = value
        counts[value['status']] += 1
        progress(f'Fragments {len(results)}/{total}; status counts {json.dumps(dict(counts), sort_keys=True)}')

    with executor_factory(max_workers=args.jobs) as executor:
        futures = {}
        for smi in fragments:
            try:
                futures[executor.submit(scorer, smi, args)] = smi
            except Exception as exc:
                record(smi, failure(fragment_configuration(smi, args), 'fragment_submission_exception', exc))
        for future in as_completed(futures):
            smi = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = failure(fragment_configuration(smi, args), 'fragment_worker_exception', exc)
            record(smi, value)
    return results


def score_records(records, args, scorer=None, executor_factory=ProcessPoolExecutor, progress=print):
    records = select_records(records, args.max_rows)
    planned, fragments = prepare_candidates(records, args)
    progress(f'Scoring {len(records)} rows; {len(planned)} candidates; {len(fragments)} unique fragments')
    scores = run_unique_fragments(fragments, args, scorer, executor_factory, progress)
    output = []
    protocol = computation_protocol(args)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for result, names in planned:
        result['computation_protocol_sha256'] = protocol_hash
        if names is not None:
            result['fragments'] = [scores[name] for name in names]  # Keep duplicate-fragment multiplicity.
            statuses = [value['status'] for value in result['fragments']]
            if all(status == 'ok' for status in statuses):
                result.update(status='ok', energy_hartree=sum(value['energy_hartree'] for value in result['fragments']))
            else:
                result.update(status='failed' if 'failed' in statuses else 'unsupported',
                              reason='one_or_more_fragments_unscored')
        output.append(result)
    summary = {'selected_rows': len(records), 'candidate_count': len(output), 'unique_fragments': len(fragments),
               'candidate_status_counts': dict(Counter(value['status'] for value in output)),
               'fragment_status_counts': dict(Counter(value['status'] for value in scores.values())),
               'selected_row_ids': [row['row_id'] for row in records],
               'k': args.k, 'max_rows': args.max_rows, 'jobs': args.jobs,
               'computation_protocol': protocol, 'computation_protocol_sha256': protocol_hash,
               'selection': 'all provided rows when max_rows=0; otherwise SHA-256(row_id) sample, preserving input order',
               'label_use': 'no correctness, recorded target, rank-error or energy-based row selection'}
    return output, summary


def write_jsonl_atomic(path, records):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+path.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'out', 'cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--xtb', default='xtb')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max-rows', type=int, default=0, help='0 scores all supplied rows; positive values select by row-ID hash')
    parser.add_argument('--conformers', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=20., help='Wall seconds for the entire fragment worker, including embedding')
    parser.add_argument('--max-heavy-atoms', type=int, default=40)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--solvent', required=True, help='gas or an xTB-supported ALPB solvent; explicit diagnostic assumption')
    args = parser.parse_args()
    if min(args.k, args.conformers, args.timeout, args.max_heavy_atoms, args.jobs) <= 0 or args.max_rows < 0:
        raise SystemExit('K, conformers, timeout, max-heavy-atoms and jobs must be positive; max-rows must be nonnegative.')
    if args.out.exists():
        raise SystemExit('Output already exists. Use a fresh output path; verified fragment caches can be reused.')
    executable = shutil.which(args.xtb)
    if not executable:
        raise SystemExit('xTB executable missing; no physical energies were computed.')
    args.xtb = str(Path(executable).resolve())
    version = subprocess.run([args.xtb, '--version'], capture_output=True, text=True, check=True, timeout=20)
    args.xtb_version = (version.stdout+'\n'+version.stderr).strip()
    args.xtb_sha256 = file_hash(args.xtb)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(args.records)
    records = load_records(args.records)
    started = time.monotonic()
    output, summary = score_records(records, args, progress=lambda text: print(text, flush=True))
    if file_hash(args.records) != source_hash:
        raise RuntimeError('Input records changed while scoring')
    write_jsonl_atomic(args.out, output)
    summary.update(source_records_sha256=source_hash, source_rows=len(records), output_sha256=file_hash(args.out),
                   elapsed_seconds=round(time.monotonic()-started, 3))
    atomic_json(args.out.with_suffix(args.out.suffix+'.manifest.json'), summary)
    print(json.dumps({'completed': True, 'rows': summary['selected_rows'], 'candidates': summary['candidate_count'],
                      'unique_fragments': summary['unique_fragments'], 'status_counts': summary['candidate_status_counts']}, sort_keys=True), flush=True)


def fragment_worker():
    request = json.load(sys.stdin)
    try:
        result = _calculate_fragment(request['smiles'], SimpleNamespace(**request['settings']))
    except Exception as exc:
        result = {'status': 'failed', 'energy_hartree': None, 'reason': 'fragment_calculation_exception',
                  'exception_type': type(exc).__name__}
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '_fragment-worker':
        fragment_worker()
    else:
        main()
