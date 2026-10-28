"""Synthetic checks for exploratory role separation, features and paired outcomes."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
spec = importlib.util.spec_from_file_location('learn_failure_patterns', Path(__file__).parents[1]/'scripts/learn_failure_patterns.py')
learner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learner)


def row(index, correct=1):
    candidates = []
    for i, (smiles, count) in enumerate(zip(('CO', 'CCO', 'CCCO'), (40, 20, 4))):
        features = [0.]*len(learner.FEATURES)
        features[0], features[1], features[2] = math.log((count+.5)/65), count/64, float(i == 1)
        candidates.append({'rank': i+1, 'smiles': smiles, 'count': count, 'correct': i == correct,
                           'parseable': True, 'features': features})
    return {'row_id': index, 'source_row_id': index, 'source_split': 'test', 'split_group': f'group-{index}',
            'sequence_id': f'test:path-{index}', 'reactant': 'C'*(index+1)+'N',
            'target': candidates[correct]['smiles'] if correct is not None else 'CCCCO',
            'target_rank': correct+1 if correct is not None else None, 'top1_correct': correct == 0,
            'sample_count': 64, 'candidates': candidates, 'rerank_eligible': True}


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(value)+'\n' for value in rows))


class FeatureAndSeparationTests(unittest.TestCase):
    def test_added_removed_bits_preserve_context_that_linear_subtraction_cancels(self):
        r0, r1, a, b = map(np.array, ([0], [1], [1], [0]))
        np.testing.assert_array_equal((a-r0)-(b-r0), (a-r1)-(b-r1))
        delta0 = learner.transformation_bits(r0, a)-learner.transformation_bits(r0, b)
        delta1 = learner.transformation_bits(r1, a)-learner.transformation_bits(r1, b)
        self.assertFalse(np.array_equal(delta0, delta1))
        np.testing.assert_array_equal(delta0, [1., 0.])
        np.testing.assert_array_equal(delta1, [0., -1.])

    def test_labels_diagnostics_and_identity_cannot_change_features_or_choices(self):
        original, changed = row(0), row(0)
        changed.update(target='unreadable target', target_rank=None, diagnostic_flags={'a': True},
                       target_no_op=True, other_recorded_target=True, split_group='another', row_id=987)
        for candidate in changed['candidates']:
            candidate['correct'] = False
        builder = learner.FeatureBuilder(10, 512)
        for family in ('simple', 'transform_fingerprint'):
            np.testing.assert_array_equal(builder.row_matrix(original, family).toarray(),
                                          builder.row_matrix(changed, family).toarray())
        config = {'alpha': 1., 'margin': 0., 'max_top1_fraction': 1.}
        scores = np.array([0., 2., 1.])
        self.assertEqual(learner.choose([original, changed], [scores, scores], config, 10), [1, 1])

    def test_unsupported_candidate_is_never_dropped_reordered_or_featurized(self):
        record = row(0, correct=None)
        record['rerank_eligible'] = False
        record['candidates'][0].update(smiles='OPAQUE[broken', parseable=False, features=['placeholder'])
        original = copy.deepcopy(record)
        builder = learner.FeatureBuilder(10, 512)
        self.assertIsNone(builder.row_matrix(record, 'simple'))
        self.assertIsNone(builder.row_matrix(record, 'transform_fingerprint'))
        config = {'alpha': 1., 'margin': 0., 'max_top1_fraction': 1.}
        choice = learner.choose([record], [None], config, 10)
        self.assertEqual(learner.outcome_rows([record], choice, 10)[0]['selected_smiles'], 'OPAQUE[broken')
        self.assertEqual(record, original)

    def test_confidence_gate_and_stable_ties_use_only_candidate_counts(self):
        record, scores = row(0), [np.array([0., 2., 1.])]
        config = {'alpha': 1., 'margin': 0., 'max_top1_fraction': .5}
        self.assertEqual(learner.choose([record], scores, config, 10), [0])
        config['max_top1_fraction'] = .7
        self.assertEqual(learner.choose([record], scores, config, 10), [1])
        self.assertEqual(learner.choose([record], [np.ones(3)], config, 10), [0])

    def test_official_test_source_is_never_relabelled_and_splits_ignore_outcomes(self):
        rows = [row(i) for i in range(60)]
        parts, manifest, unused = learner.make_splits(rows, 'digest')
        self.assertEqual(sum(len(part) for part in parts.values()), len(rows))
        self.assertTrue(all(item['source_split'] == 'test' for part in parts.values() for item in part))
        self.assertEqual(manifest['status'], 'exploratory_not_untouched')
        changed = copy.deepcopy(rows)
        for item in changed:
            item['top1_correct'] = True
        self.assertEqual(manifest['group_assignments'], learner.make_splits(changed, 'digest')[1]['group_assignments'])
        rows[0]['source_split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'source_split'):
            learner.validate_records(rows)

    def test_input_and_pathway_links_cannot_cross_frozen_groups(self):
        a, b = row(0), row(1)
        a['reactant'], b['reactant'] = 'CCO', 'OCC'
        with self.assertRaisesRegex(ValueError, 'inconsistent frozen split_group'):
            learner.validate_records([a, b])
        a, b = row(0), row(1)
        b['sequence_id'] = a['sequence_id']
        with self.assertRaisesRegex(ValueError, 'inconsistent frozen split_group'):
            learner.validate_records([a, b])

    def test_input_weight_is_equal_with_different_numbers_of_negatives(self):
        rows = [row(i) for i in range(20)]
        for item in rows[::2]:
            item['candidates'] = item['candidates'][:2]
        with patch.object(learner, 'make_pipeline') as pipeline:
            learner.fit_models(rows, learner.FeatureBuilder(), 'simple')
        weights = pipeline.return_value.fit.call_args.kwargs['logisticregression__sample_weight']
        offset = 0
        for item in rows:
            length = 2*(len(item['candidates'])-1)
            self.assertAlmostEqual(weights[offset:offset+length].sum(), 1.)
            offset += length

    def test_ambiguity_policy_uses_only_alternative_targets_observed_in_fit(self):
        rows = [row(i) for i in range(20)]
        alternative = row(20, correct=0)
        alternative['reactant'] = rows[0]['reactant']
        alternative['split_group'] = rows[0]['split_group']
        rows.append(alternative)
        with patch.object(learner, 'make_pipeline'):
            _, strict = learner.fit_models(rows, learner.FeatureBuilder(), 'simple')
            _, filtered = learner.fit_models(rows, learner.FeatureBuilder(), 'transform_fingerprint')
        self.assertEqual(strict['contradictory_negative_pairs_skipped'], 0)
        self.assertEqual(filtered['contradictory_negative_pairs_skipped'], 2)
        self.assertEqual(filtered['fit_reactants_with_multiple_observed_targets'], 1)
        self.assertEqual(strict['candidate_pairs']-filtered['candidate_pairs'], 2)


class ExploratoryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records = self.root/'records.jsonl'
        self.rows = [row(i) for i in range(24)]+[row(i) for i in range(30, 42)]+[
            row(i, correct=i % 2) for i in range(50, 62)]
        unsupported = row(62, correct=None)
        unsupported['rerank_eligible'] = False
        unsupported['candidates'][0].update(smiles='OPAQUE[broken', parseable=False, features=['placeholder'])
        absent, empty = row(63, correct=None), row(64, correct=None)
        empty['candidates'] = []; empty['rerank_eligible'] = False
        self.rows += [unsupported, absent, empty]
        write_rows(self.records, self.rows)
        self.manifest = self.root/'input-split.json'
        assignments = {item['split_group']: 'fit' if item['row_id'] < 24 else
                       'tune' if item['row_id'] < 50 else 'assessment' for item in self.rows}
        self.manifest.write_text(json.dumps({'group_assignments': assignments}))
        self.fit_args = SimpleNamespace(records=self.records, out=self.root/'model', k=10,
                                       fingerprint_bits=512, no_fingerprints=False, seed=20261028,
                                       split_manifest=self.manifest)
        with redirect_stdout(io.StringIO()):
            self.receipt = learner.fit_tune(self.fit_args)
        self.assess_args = SimpleNamespace(records=self.records, model=self.root/'model/frozen.joblib',
                                          out=self.root/'assessment')

    def tearDown(self):
        self.temp.cleanup()

    def test_counts_new_errors_without_claiming_untouched_or_resume_permission(self):
        with redirect_stdout(io.StringIO()):
            report = learner.assess(self.assess_args)
        self.assertEqual(report['rows'], 15)
        self.assertEqual(report['status'], 'exploratory_not_untouched')
        self.assertFalse(report['benchmark_resume_allowed'])
        self.assertEqual(report['methods']['simple']['corrected'], 6)
        self.assertEqual(report['methods']['simple']['harmed'], 6)
        self.assertEqual(report['methods']['simple']['changed_wrong_to_wrong'], 1)
        self.assertEqual(report['methods']['simple']['target_absent_from_top_k'], 3)
        self.assertEqual(report['methods']['simple']['covered_rows'], 13)
        self.assertEqual(report['methods']['simple']['net_accuracy_change_pp'], 0.)
        decisions = [json.loads(line) for line in (self.assess_args.out/'choices.jsonl').read_text().splitlines()]
        opaque = next(item for item in decisions if item['row_id'] == 62)
        self.assertEqual(opaque['methods']['transform_fingerprint']['selected_smiles'], 'OPAQUE[broken')
        self.assertTrue(all(item['source_split'] == 'test' for item in decisions))

    def test_frozen_assessment_replay_detects_artifact_mutation(self):
        with redirect_stdout(io.StringIO()):
            first = learner.assess(self.assess_args)
            second = learner.assess(self.assess_args)
        self.assertEqual(first, second)
        with (self.assess_args.out/'choices.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            learner.assess(self.assess_args)

    def test_source_change_after_freezing_is_rejected(self):
        self.rows[-2]['candidates'][0]['features'][2] = 9.
        write_rows(self.records, self.rows)
        with self.assertRaisesRegex(ValueError, 'differs from the frozen diagnostic cohort'):
            learner.assess(self.assess_args)


if __name__ == '__main__':
    unittest.main()
