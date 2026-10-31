import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import full_validation_reranker as full
from prepare_full_validation_test20k import select_test, write_parts
from test_targeted_reranker import row


class FullValidationRerankerTests(unittest.TestCase):
    def test_test_sampling_is_fixed_outcome_blind_and_excludes_ineligible_rows(self):
        rows = [dict(source_row_id=i, fresh_group_disjoint=i >= 10,
                     target=f'label-{i}', correct=i % 2 == 0) for i in range(100)]
        first = [r['source_row_id'] for r in select_test(rows, 20)]
        changed = [dict(r, target='replaced', correct=not r['correct']) for r in reversed(rows)]
        self.assertEqual(first, [r['source_row_id'] for r in select_test(changed, 20)])
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(i >= 10 for i in first))

    def test_noncontiguous_inference_parts_keep_exact_source_membership(self):
        import hashlib
        lines = [f'input-{i}>>target-{i}|{i}\n'.encode() for i in range(8)]
        rows = [dict(source_row_id=i, source_split='val',
                     source_line_sha256=hashlib.sha256(lines[i]).hexdigest()) for i in (1, 4, 7)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            source = root / 'source.txt'; source.write_bytes(b''.join(lines))
            parts = write_parts(source, rows, root, 'val', size=2)
            saved = [r['source_row_id'] for name in parts
                     for r in json.loads((root / f'{name}-rows.json').read_text())]
            self.assertEqual(saved, [1, 4, 7])
            self.assertEqual(sum(p['rows'] for p in parts.values()), 3)

    def test_fitting_uses_only_validation_and_refits_every_row_from_scratch(self):
        rows = [dict(row(i, split='val'), cv_fold=i % 5) for i in range(30)]
        original_fit = full.core.fit_models
        calls = []
        def observe(fit_rows, matrices, family):
            calls.append([r['source_row_id'] for r in fit_rows])
            return original_fit(fit_rows, matrices, family)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {'training_rows': 30, 'cv_folds': 5, 'test_rows': 3,
                        'manifest_sha256': {'test-rows.json': 'test-membership'}}
            (root / 'protocol.json').write_text(json.dumps(protocol))
            with patch.object(full, 'experiment', return_value=protocol), \
                    patch.object(full, 'load_complete', return_value=(rows, {})) as loader, \
                    patch.object(full.core, 'fit_models', side_effect=observe), \
                    patch.object(full.joblib, 'load', side_effect=AssertionError('Old weights must not be loaded')), \
                    redirect_stdout(io.StringIO()):
                full.fit(SimpleNamespace(experiment=root))
            loader.assert_called_once_with(root, 'val')
            self.assertEqual(len(calls), 6)
            for fold in range(5):
                self.assertEqual(set(calls[fold]), {i for i in range(30) if i % 5 != fold})
            self.assertEqual(calls[-1], list(range(30)))
            receipt = json.loads((root / 'model' / 'frozen.json').read_text())
            self.assertEqual(receipt['training_rows'], 30)
            self.assertFalse(receipt['test_used_for_training_or_selection'])
            model = joblib.load(root / 'model' / 'frozen.joblib')
            self.assertEqual(len(model['features']), 97)
            self.assertEqual(model['model'].named_steps['logisticregression'].coef_.shape, (1, 97))
            self.assertEqual(receipt['cv_selected']['summary']['rows'], 30)
            test_rows = [row(31, split='test', correct=0), row(32, split='test', correct=1),
                         row(33, split='test', correct=None)]
            with patch.object(full, 'experiment', return_value=protocol), \
                    patch.object(full, 'load_complete', return_value=(test_rows, {})) as evaluator, \
                    redirect_stdout(io.StringIO()):
                report = full.evaluate(SimpleNamespace(experiment=root))
                repeated = full.evaluate(SimpleNamespace(experiment=root))
            self.assertEqual(evaluator.call_args.args, (root, 'test'))
            self.assertEqual(report, repeated)
            self.assertEqual(report['rows'], 3)
            self.assertEqual(report['frequency']['correct'], 1)
            self.assertEqual(report['new_reranker']['oracle_top_k_correct'], 2)
            self.assertEqual(report['new_reranker']['target_absent_from_top_k'], 1)

    def test_membership_rejects_missing_duplicate_foreign_and_changed_rows(self):
        rows = [dict(row(i, split='val'), reactant_key=f'input-{i}') for i in range(3)]
        metadata = [dict(r, candidate_origin='new_generation', cv_fold=i)
                    for i, r in enumerate(rows)]
        for bad in (rows[:2], [rows[0], rows[0], rows[2]],
                    [dict(rows[0], source_row_id=99), *rows[1:]],
                    [dict(rows[0], source_line_sha256='changed'), *rows[1:]]):
            with self.subTest(bad=bad[0]['source_row_id']), self.assertRaises(ValueError):
                full.validate_membership(copy.deepcopy(bad), metadata, 'val')

    def test_fold_leakage_through_same_input_is_rejected(self):
        rows = [dict(row(i, split='val'), cv_fold=i % 5) for i in range(30)]
        rows[1]['reactant'] = rows[0]['reactant']
        matrices, _ = full.core.prepare(rows)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            full.cross_validate(rows, matrices)


if __name__ == '__main__':
    unittest.main()
