"""Regression tests for data separation and paired reranking evaluation."""
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
spec = importlib.util.spec_from_file_location('pilot_reranker', Path(__file__).parents[1]/'scripts/pilot_reranker.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def row(index=0, split='test', correct=1, reactant=None, group=None):
    smiles = [pilot.canonical(smi) for smi in ('CO', 'CCO', 'CCCO')]
    counts = (40, 20, 4)
    candidates = []
    for i, (smi, count) in enumerate(zip(smiles, counts)):
        features = [0.]*len(pilot.FEATURES)
        features[0] = math.log((count+.5)/65)
        features[1] = count/64
        features[2] = float(i == 1)
        candidates.append({'rank': i+1, 'smiles': smi, 'count': count,
                           'correct': i == correct, 'features': features})
    return {'row_id': index, 'source_row_id': index, 'source_split': split,
            'source_line_sha256': hashlib.sha256(f'{split}:{index}'.encode()).hexdigest(),
            'sequence_id': f'{split}:p{index}', 'split_group': group or f'{split}:g{index}',
            'reactant': pilot.canonical(reactant or 'C'*(index+1)+'N'),
            'target': smiles[correct] if correct is not None else pilot.canonical('CCCCO'),
            'sample_count': 64, 'top1_correct': correct == 0, 'candidates': candidates}


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(item)+'\n' for item in rows))


class OutcomeTests(unittest.TestCase):
    def test_counts_both_corrections_and_new_errors_and_all_rows(self):
        rows = [row(0, correct=1), row(1, correct=0), row(2, correct=0), row(3, correct=None), row(4, correct=None)]
        rows[-1]['candidates'] = []
        outcomes = pilot.outcome_rows(rows, [1, 1, 0, 1, None], 5)
        summary = pilot.summarize(outcomes)
        self.assertEqual(summary['rows'], 5)
        self.assertEqual(summary['corrected'], 1)
        self.assertEqual(summary['harmed'], 1)
        self.assertEqual(summary['unchanged_correct'], 1)
        self.assertEqual(summary['unchanged_wrong'], 2)
        self.assertEqual(summary['changed_wrong_to_wrong'], 1)
        self.assertEqual(summary['net_correct'], 0)
        self.assertEqual(summary['target_absent_from_top_k'], 2)
        self.assertEqual(summary['accuracy'], .4)

    def test_stable_ties_and_strict_switch_margin(self):
        self.assertEqual(pilot.stable_choice(np.array([1., 1., 0.])), 0)
        self.assertEqual(pilot.stable_choice(np.array([0., 2., 2.])), 1)
        self.assertEqual(pilot.stable_choice(np.array([0., 2., 2.]), margin=2.), 0)
        self.assertIsNone(pilot.stable_choice(np.array([])))

    def test_energy_requires_complete_candidate_coverage(self):
        record = row()
        candidates = record['candidates']
        scores = {(str(record['row_id']), c['smiles']): {'status': 'ok', 'energy_hartree': -1.-i}
                  for i, c in enumerate(candidates[:2])}
        choices, covered = pilot.energy_choices([record], scores, 5)
        self.assertEqual((choices, covered), ([0], [False]))
        scores[(str(record['row_id']), candidates[2]['smiles'])] = {'status': 'unsupported'}
        self.assertEqual(pilot.energy_choices([record], scores, 5), ([0], [False]))
        scores[(str(record['row_id']), candidates[2]['smiles'])] = {'status': 'ok', 'energy_hartree': -.5}
        self.assertEqual(pilot.energy_choices([record], scores, 5), ([1], [True]))
        self.assertEqual(pilot.energy_choices([record], scores, 5, weight=0.), ([0], [True]))

    def test_group_bootstrap_does_not_treat_dependent_rows_as_independent(self):
        outcomes = [{'split_group': 'large', 'baseline_correct': False, 'correct': True} for _ in range(100)]
        outcomes.append({'split_group': 'small', 'baseline_correct': True, 'correct': False})
        bootstrap = pilot.cluster_bootstrap(outcomes)
        self.assertEqual(bootstrap['group_count'], 2)
        self.assertEqual(bootstrap['lower'], -1.)
        self.assertEqual(bootstrap['upper'], 1.)
        self.assertAlmostEqual(bootstrap['point_estimate'], 99/101)
        self.assertEqual(bootstrap, pilot.cluster_bootstrap(list(reversed(outcomes))))

    def test_gate_rejects_tiny_or_negative_results(self):
        summary = {'corrected': 10, 'harmed': 0, 'net_accuracy_delta': .1}
        interval = {'lower': .05, 'group_count': 40}
        self.assertEqual(pilot.credible_benefit(summary, interval)['status'], 'tiny')
        summary.update(corrected=30, harmed=0)
        self.assertTrue(pilot.credible_benefit(summary, interval)['passed'])
        summary.update(corrected=10, harmed=30, net_accuracy_delta=-.1)
        self.assertEqual(pilot.credible_benefit(summary, interval)['status'], 'negative')

    def test_validation_ties_prefer_fewer_changes(self):
        safer = {'correct': 20, 'changed': 2, 'harmed': 0}
        busier = {'correct': 20, 'changed': 5, 'harmed': 1}
        self.assertGreater(pilot.tuning_key(safer), pilot.tuning_key(busier))


class SeparationTests(unittest.TestCase):
    def test_official_split_is_enforced(self):
        with self.assertRaisesRegex(ValueError, 'source_split=train'):
            pilot.validate_records([row(split='test')], 'train')

    def test_reactant_leakage_uses_canonical_equivalence(self):
        fit = row(0, 'train', reactant='CCO')
        test = row(1, 'test', reactant='OCC')
        with self.assertRaisesRegex(ValueError, 'shared reactants'):
            pilot.assert_disjoint(pilot.source_signature([fit]), pilot.source_signature([test]), 'Training/test')

    def test_pathway_and_group_overlap_are_both_rejected(self):
        fit, test = row(0, 'train'), row(3, 'test')
        test['sequence_id'] = fit['sequence_id']
        with self.assertRaisesRegex(ValueError, 'shared sequences'):
            pilot.assert_disjoint(pilot.source_signature([fit]), pilot.source_signature([test]), 'Training/test')
        test['sequence_id'] = 'test:other'
        test['split_group'] = fit['split_group']
        with self.assertRaisesRegex(ValueError, 'shared groups'):
            pilot.assert_disjoint(pilot.source_signature([fit]), pilot.source_signature([test]), 'Training/test')

    def test_generic_sequence_labels_are_not_shared_pathway_ids(self):
        fit, test = row(0, 'train'), row(3, 'test')
        fit['sequence_id'] = test['sequence_id'] = 'PM'
        pilot.assert_disjoint(pilot.source_signature([fit]), pilot.source_signature([test]), 'Training/test')

    def test_connected_rows_cannot_be_split_into_independent_bootstrap_groups(self):
        a, b = row(0), row(1)
        b['sequence_id'] = a['sequence_id']
        with self.assertRaisesRegex(ValueError, 'different split_group'):
            pilot.validate_records([a, b], 'test')
        b['split_group'] = a['split_group']
        pilot.validate_records([a, b], 'test')


class FrozenWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fit, self.tune, self.test = (self.root/f'{name}.jsonl' for name in ('fit', 'tune', 'test'))
        write_rows(self.fit, [row(i, 'train') for i in range(24)])
        write_rows(self.tune, [row(i, 'val') for i in range(30, 42)])
        write_rows(self.test, [row(i, 'test', correct=i % 2) for i in range(50, 62)])
        self.manifest = self.root/'holdout-manifest.json'
        manifest = []
        for record in [row(i, 'test', correct=i % 2) for i in range(50, 62)]:
            entry = {key: record[key] for key in ('row_id', 'source_row_id', 'source_split',
                                                 'split_group', 'sequence_id', 'source_line_sha256')}
            entry['reactant_key'] = hashlib.sha256(record['reactant'].encode()).hexdigest()
            manifest.append(entry)
        self.manifest.write_text(json.dumps(manifest))
        self.fit_args = SimpleNamespace(fit=self.fit, tune=self.tune, out=self.root/'model', k=5,
                                       fit_energies=None, tune_energies=None, holdout_manifest=self.manifest)
        with redirect_stdout(io.StringIO()):
            self.receipt = pilot.fit_tune(self.fit_args)
        self.eval_args = SimpleNamespace(records=self.test, model=self.root/'model/frozen.joblib',
                                        out=self.root/'evaluation', energies=None, holdout_manifest=self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def test_fit_selects_only_on_validation_and_holdout_counts_regressions(self):
        self.assertEqual(self.receipt['primary_method'], 'learned')
        self.assertEqual(self.receipt['training']['candidate_pairs'], 48)
        with redirect_stdout(io.StringIO()):
            report = pilot.evaluate(self.eval_args)
        self.assertEqual(report['rows'], 12)
        self.assertEqual(report['methods']['frequency']['correct'], 6)
        self.assertEqual(report['methods']['learned']['corrected'], 6)
        self.assertEqual(report['methods']['learned']['harmed'], 6)
        self.assertEqual(report['methods']['learned']['net_correct'], 0)
        self.assertFalse(report['benchmark_resume_gate']['passed'])

    def test_exact_replay_reads_artifacts_but_changed_cohort_is_rejected(self):
        with redirect_stdout(io.StringIO()):
            first = pilot.evaluate(self.eval_args)
            replay = pilot.evaluate(self.eval_args)
        self.assertEqual(first, replay)
        write_rows(self.test, [row(i, 'test') for i in range(50, 63)])
        with self.assertRaisesRegex(ValueError, 'membership differs'):
            pilot.evaluate(self.eval_args)

    def test_model_or_holdout_manifest_mutation_fails(self):
        self.manifest.write_text('{"different":true}')
        with self.assertRaisesRegex(ValueError, 'precommitted holdout manifest'):
            pilot.evaluate(self.eval_args)
        self.eval_args.model.chmod(0o644)
        with self.eval_args.model.open('ab') as stream:
            stream.write(b'mutation')
        with self.assertRaisesRegex(ValueError, 'Frozen model changed'):
            pilot.evaluate(self.eval_args)

    def test_evaluation_checks_overlap_against_tuning_too(self):
        contaminated = row(50, 'test')
        contaminated['reactant'] = row(30, 'val')['reactant']
        write_rows(self.test, [contaminated])
        # Membership validation rejects even a single contaminated replacement.
        with self.assertRaisesRegex(ValueError, 'membership differs'):
            pilot.evaluate(self.eval_args)

    def test_replay_checks_saved_artifact_hashes(self):
        with redirect_stdout(io.StringIO()):
            pilot.evaluate(self.eval_args)
        with (self.eval_args.out/'decisions.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'evaluation artifacts changed'):
            pilot.evaluate(self.eval_args)

    def test_matching_ids_cannot_hide_changed_source_identity(self):
        records = [row(i, 'test', correct=i % 2) for i in range(50, 62)]
        records[0]['source_row_id'] = 9999
        write_rows(self.test, records)
        with self.assertRaisesRegex(ValueError, 'source_row_id differs'):
            pilot.evaluate(self.eval_args)

    def test_fitted_output_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            pilot.fit_tune(self.fit_args)


if __name__ == '__main__':
    unittest.main()
