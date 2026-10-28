"""Physical pilot orchestration tests; no xTB executable or external process runs."""
from concurrent.futures import Future
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
spec = importlib.util.spec_from_file_location('score_xtb_candidates', Path(__file__).parents[1]/'scripts/score_xtb_candidates.py')
xtb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xtb)


class ImmediateExecutor:
    """Future-compatible test executor; submission results can complete in any order."""
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submissions = []

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass

    def submit(self, scorer, smiles, args):
        self.submissions.append(smiles)
        future = Future()
        try:
            future.set_result(scorer(smiles, args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def record(row_id, reactant, candidates):
    return {'row_id': row_id, 'reactant': reactant, 'target': candidates[0], 'top1_correct': True,
            'candidates': [{'rank': index+1, 'smiles': smi, 'correct': index == 0}
                           for index, smi in enumerate(candidates)]}


class PhysicalPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(k=5, max_rows=0, conformers=1, timeout=20., max_heavy_atoms=40,
                                    jobs=4, solvent='gas', xtb='/not/executed/xtb', xtb_version='mock-6.7.1',
                                    xtb_sha256='mock-binary-hash', cache=Path(self.temporary.name)/'cache')
        self.executors, self.messages = [], []

    def tearDown(self):
        self.temporary.cleanup()

    def executor(self, **kwargs):
        executor = ImmediateExecutor(**kwargs)
        self.executors.append(executor)
        return executor

    def score(self, records, scorer):
        return xtb.score_records(records, self.args, scorer=scorer,
                                 executor_factory=self.executor, progress=self.messages.append)

    def test_deduplication_preserves_fragment_multiplicity(self):
        rows = [record(9, 'CO.CO', ['OC.CO']), record(2, 'CO', ['CO'])]
        scorer = Mock(return_value={'status': 'ok', 'energy_hartree': -3.})
        output, summary = self.score(rows, scorer)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(len(self.executors[0].submissions), 1)
        self.assertEqual(self.executors[0].max_workers, 4)
        self.assertEqual(summary['unique_fragments'], 1)
        self.assertEqual([item['row_id'] for item in output], [9, 2])
        self.assertEqual([item['energy_hartree'] for item in output], [-6., -3.])

    def test_output_row_and_candidate_order_does_not_follow_completion_order(self):
        rows = [record(7, 'CCO', ['COC', 'CCO']), record(2, 'CCO', ['CCO', 'COC'])]
        scores = {xtb.canonical('CCO'): -2., xtb.canonical('COC'): -1.}
        output, summary = self.score(rows, lambda smiles, unused: {'status': 'ok', 'energy_hartree': scores[smiles]})
        self.assertEqual(len(self.executors[0].submissions), 2)
        self.assertEqual([(item['row_id'], item['smiles']) for item in output],
                         [(7, 'COC'), (7, 'CCO'), (2, 'CCO'), (2, 'COC')])
        self.assertEqual([item['energy_hartree'] for item in output], [-1., -2., -2., -1.])
        self.assertEqual(summary['candidate_status_counts'], {'ok': 4})

    def test_fragment_exception_fails_affected_candidates_without_crashing_cohort(self):
        def scorer(smiles, unused):
            if smiles == xtb.canonical('COC'):
                raise RuntimeError('simulated fragment crash')
            return {'status': 'ok', 'energy_hartree': -2.}
        output, summary = self.score([record(0, 'CCO', ['COC', 'CCO'])], scorer)
        self.assertEqual([item['status'] for item in output], ['failed', 'ok'])
        self.assertEqual(output[0]['fragments'][0]['exception_type'], 'RuntimeError')
        self.assertIsNone(output[0]['energy_hartree'])
        self.assertEqual(summary['fragment_status_counts'], {'failed': 1, 'ok': 1})

    def test_nonfinite_fragment_result_is_failed(self):
        output, _ = self.score([record(0, 'CO', ['CO'])],
                               lambda *unused: {'status': 'ok', 'energy_hartree': float('nan')})
        self.assertEqual(output[0]['status'], 'failed')
        self.assertEqual(output[0]['fragments'][0]['reason'], 'invalid_fragment_result')

    def test_inventory_mismatch_never_schedules_a_fragment(self):
        scorer = Mock(return_value={'status': 'ok', 'energy_hartree': -1.})
        output, summary = self.score([record(0, 'CCO', ['C'])], scorer)
        scorer.assert_not_called()
        self.assertEqual(summary['unique_fragments'], 0)
        self.assertEqual(output[0]['status'], 'unsupported')
        self.assertEqual(output[0]['reason'], 'element_isotope_or_net_charge_mismatch')

    def test_max_rows_zero_means_all_and_sampling_does_not_read_labels(self):
        rows = [record(index, 'CO', ['CO']) for index in range(20)]
        self.assertEqual(xtb.select_records(rows, 0), rows)
        selected = xtb.select_records(rows, 7)
        self.assertEqual(len(selected), 7)
        self.assertEqual([r['row_id'] for r in selected], sorted(r['row_id'] for r in selected))
        flipped = copy.deepcopy(rows)
        for item in flipped:
            item['target'] = 'not consulted'
            item['top1_correct'] = False
            item['candidates'][0]['correct'] = False
        self.assertEqual([r['row_id'] for r in selected], [r['row_id'] for r in xtb.select_records(flipped, 7)])
        self.score(flipped[:1], lambda *unused: {'status': 'ok', 'energy_hartree': -1.})
        self.assertFalse(any('correct' in message for message in self.messages))

    def test_cache_separates_timeout_size_and_version_and_repairs_invalid_json(self):
        smiles = xtb.canonical('CO')
        result = {'status': 'failed', 'energy_hartree': None, 'reason': 'fragment_wall_timeout'}
        with patch.object(xtb, '_compute_bounded', return_value=result) as compute:
            xtb.fragment_energy(smiles, self.args)
            xtb.fragment_energy(smiles, self.args)
            self.assertEqual(compute.call_count, 1)
            self.args.timeout = 30.
            xtb.fragment_energy(smiles, self.args)
            self.args.max_heavy_atoms = 60
            xtb.fragment_energy(smiles, self.args)
            self.args.xtb_version = 'different-version'
            xtb.fragment_energy(smiles, self.args)
            self.assertEqual(compute.call_count, 4)
            cache = xtb.cache_path(xtb.fragment_configuration(smiles, self.args), self.args)
            cache.write_text('{broken')
            xtb.fragment_energy(smiles, self.args)
            self.assertEqual(compute.call_count, 5)
        self.assertEqual(len(list(self.args.cache.glob('*.json'))), 4)
        self.assertFalse(list(self.args.cache.glob('*.tmp')))
        self.assertEqual(json.loads(cache.read_text())['computation_protocol']['timeout_seconds'], 30.)

    def test_fragment_timeout_kills_the_entire_child_process_group(self):
        process = Mock(pid=987654)
        process.communicate.side_effect = [subprocess.TimeoutExpired('worker', 20), ('', '')]
        with patch.object(xtb.subprocess, 'Popen', return_value=process) as launch, \
                patch.object(xtb.os, 'killpg') as kill:
            result = xtb._compute_bounded(xtb.canonical('CO'), self.args)
        self.assertEqual(result['reason'], 'fragment_wall_timeout')
        kill.assert_called_once_with(987654, xtb.signal.SIGKILL)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertEqual(process.communicate.call_args_list[0].kwargs['timeout'], 20.)

    def test_unsupported_radical_isotope_and_size_need_no_xtb_call(self):
        with patch.object(xtb.subprocess, 'run') as run:
            radical = xtb._calculate_fragment('[CH3]', self.args)
            isotope = xtb._calculate_fragment('[13CH4]', self.args)
            large = xtb._calculate_fragment('C'*41, self.args)
        run.assert_not_called()
        self.assertEqual(radical['reason'], 'unassigned_spin_radical')
        self.assertEqual(isotope['reason'], 'isotope_not_supported_by_this_xyz_protocol')
        self.assertEqual(large['reason'], 'size_limit')


if __name__ == '__main__':
    unittest.main()
