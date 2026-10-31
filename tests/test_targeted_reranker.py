"""Synthetic chemistry, leakage, abstention and frozen-audit checks (no user fitting)."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import targeted_reranker as pilot


def row(index=0, split='train', correct=1, smiles=None, reactant=None, counts=None):
    chain = 'C'*(index+1)
    reactant = reactant or chain+'[O-].Cl'
    smiles = smiles or [reactant, chain+'O.[Cl-]', chain+'[OH2+].[Cl-].[Cl-]']
    counts = counts or [32, 30, 2][:len(smiles)]
    candidates = [{'rank': i+1, 'smiles': value, 'count': n, 'correct': i == correct,
                   'parseable': True, 'features': ['untrusted and deliberately unused']}
                  for i, (value, n) in enumerate(zip(smiles, counts))]
    return {'row_id': index, 'source_row_id': index, 'source_split': split,
            'source_line_sha256': hashlib.sha256(f'{split}:{index}'.encode()).hexdigest(),
            'sequence_id': f'{split}:sequence-{index}', 'split_group': f'group-{split}-{index}',
            'selection_cohort': 'random', 'reactant': reactant,
            'target': candidates[correct]['smiles'] if correct is not None else 'absent target',
            'sample_count': 64, 'top1_correct': correct == 0, 'candidates': candidates,
            'rerank_eligible': True}


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))


def manifest_for(rows):
    result = []
    for r in rows:
        entry = {key: r[key] for key in ('row_id', 'source_row_id', 'source_split', 'split_group',
                                        'sequence_id', 'source_line_sha256', 'selection_cohort')}
        entry['reactant_key'] = hashlib.sha256(pilot.input_key(r['reactant']).encode()).hexdigest()
        result.append(entry)
    return result


class ChemistryTests(unittest.TestCase):
    def test_specific_proton_acceptor_environments_are_distinct(self):
        counts = pilot.describe('CN.CC(N)=O.c1ccncc1.C[O-].CC(=O)[O-].CO.O.COC')['counts']
        self.assertEqual(counts['neutral_amine_nh'], 1)
        self.assertEqual(counts['amide_nh'], 1)
        self.assertEqual(counts['aromatic_n_acceptor'], 1)
        self.assertEqual(counts['alkoxide_o'], 1)
        self.assertEqual(counts['carboxylate_o'], 1)
        self.assertEqual(counts['alcohol_oh'], 1)
        self.assertEqual(counts['water_o'], 1)
        self.assertEqual(counts['ether_o'], 1)
        self.assertEqual(pilot.describe('CC(=O)O')['counts']['ester_carbonyl'], 0)
        self.assertEqual(pilot.describe('CC(=O)OC')['counts']['ester_carbonyl'], 1)

    def test_explicit_h_and_atom_mapping_do_not_change_environment_counts(self):
        plain = pilot.describe('CO')
        explicit = pilot.describe('[H:8][O:4][C:7]([H:9])([H:2])[H:1]')
        self.assertEqual(plain['counts'], explicit['counts'])
        self.assertEqual(plain['heavy_key'], explicit['heavy_key'])
        first = row(smiles=['[CH3:2][OH:4]', '[CH2:3]=[O:7]'], reactant='[CH3:5][OH:6]', counts=[32, 30])
        second = copy.deepcopy(first)
        second['reactant'] = '[OH:88][CH3:99]'
        second['candidates'][0]['smiles'] = '[OH:12][CH3:29]'
        second['candidates'][1]['smiles'] = '[O:17]=[CH2:22]'
        np.testing.assert_array_equal(pilot.features(first)[0], pilot.features(second)[0])

    def test_valence_flags_report_unsupported_carbon_and_supported_charge_pair(self):
        invalid = pilot.describe('C(C)(C)(C)(C)C')
        self.assertFalse(invalid['supported'])
        self.assertEqual(invalid['flags']['carbon_neighbors_gt4'], 1)
        self.assertEqual(invalid['flags']['carbon_bond_valence_gt4'], 1)
        pair = pilot.describe('[C+2].[Al-2](Cl)(Cl)Cl')
        self.assertTrue(pair['supported'])
        self.assertEqual(pair['flags']['carbon_aluminum_charge_pair'], 1)
        self.assertEqual(pilot.describe('[NH4+].[Cl-]')['flags']['neutral_n_valence_gt3'], 0)

    def test_hcl_alkoxide_signal_is_conditional_and_can_score_target_absent_rows(self):
        record = row(correct=None)
        matrix, support = pilot.features(record)
        self.assertEqual(support['reason'], 'supported')
        column = pilot.TARGETED_NAMES.index('hcl_alkoxide_neutralization')
        np.testing.assert_array_equal(matrix[:, column], [0, 1, 0])
        choice = pilot.choose([record], [matrix], [pilot.score_rule(matrix)],
                              {'alpha': 1., 'margin': 0.}, 'chemistry_rule')
        self.assertEqual(choice, [1])
        self.assertTrue(pilot.outcome_rows([record], choice, 10)[0]['changed_wrong_to_wrong'])
        terminal = row(smiles=['CCO', 'CC=O'], reactant='CCO', counts=[32, 30], correct=0)
        matrix, _ = pilot.features(terminal)
        self.assertEqual(pilot.choose([terminal], [matrix], [pilot.score_rule(matrix)],
                                     {'alpha': 1., 'margin': 0.}, 'chemistry_rule'), [0])

    def test_ring_closure_context_respects_fragment_separation(self):
        same = pilot.describe('NCCCc1cccc(Cl)n1')
        separate = pilot.describe('NCCC.c1cccc(Cl)n1')
        self.assertTrue(same['supported'] and separate['supported'])
        self.assertEqual(sum(same['context'][4:]), 1)
        self.assertEqual(sum(separate['context'][4:]), 0)

    def test_labels_identity_graphs_and_supplied_features_cannot_affect_scoring(self):
        original = row()
        altered = copy.deepcopy(original)
        altered.update(target='DO NOT READ', top1_correct=True, target_rank=1,
                       target_no_op=True, diagnostic_flags={'reference_supported': True},
                       row_id='unrelated', source_split='test', split_group='other', sequence_id='test:new')
        for c in altered['candidates']:
            c.update(correct=not c['correct'], features=[float('nan')], no_op=True,
                     properties={'abs_charge': 999}, reference_graph_distance=0)
        np.testing.assert_array_equal(pilot.features(original)[0], pilot.features(altered)[0])
        minimal = {k: original[k] for k in ('reactant', 'sample_count', 'rerank_eligible')}
        minimal['candidates'] = [{k: c[k] for k in ('smiles', 'count')} for c in original['candidates']]
        np.testing.assert_array_equal(pilot.features(original)[0], pilot.features(minimal)[0])

    def test_unsupported_and_empty_candidates_preserve_raw_baseline(self):
        record = row(correct=None)
        record['candidates'][0].update(smiles='OPAQUE[broken', parseable=False)
        prior = copy.deepcopy(record)
        matrix, support = pilot.features(record)
        self.assertIsNone(matrix)
        self.assertEqual(support['unsupported_candidate_ranks'], [1])
        self.assertEqual(pilot.choose([record], [None], [np.array([0, 100, 1])],
                                     {'alpha': 1., 'margin': 0.}, pilot.PRIMARY), [0])
        self.assertEqual(record, prior)
        supported = row()
        supported['rerank_eligible'] = False
        self.assertIsNone(pilot.features(supported)[0])
        empty = row(correct=None)
        empty['candidates'] = []
        self.assertEqual(pilot.features(empty), (None, {'reason': 'no_saved_candidates'}))
        self.assertEqual(pilot.baseline([record, empty]), [0, None])

    def test_global_ablation_has_only_global_features(self):
        matrix, _ = pilot.features(row())
        self.assertEqual(matrix.shape[1], len(pilot.TARGETED_NAMES))
        global_matrix = pilot.family_matrix(matrix, 'learned_global')
        self.assertEqual(global_matrix.shape[1], 13)
        np.testing.assert_array_equal(global_matrix, matrix[:, :13])
        self.assertEqual(len(pilot.TARGETED_NAMES), len(set(pilot.TARGETED_NAMES)))


class FittingAndOutcomeTests(unittest.TestCase):
    def test_symmetric_pairs_have_equal_input_weight_and_include_correct_controls(self):
        rows = [row(0, correct=0), row(1, correct=1), row(2, correct=None), row(3)]
        rows[1]['candidates'] = rows[1]['candidates'][:2]
        matrices, _ = pilot.prepare(rows)
        matrices[-1] = None
        x, y, weights, info = pilot.pairwise_data(rows, matrices, pilot.PRIMARY)
        self.assertEqual(info['paired_rows'], 2)
        self.assertEqual(info['candidate_pairs'], 3)
        self.assertEqual(info['paired_groups'], 2)
        self.assertAlmostEqual(sum(weights[:4]), 1)
        self.assertAlmostEqual(sum(weights[4:]), 1)
        np.testing.assert_array_equal(x[::2], -x[1::2])
        np.testing.assert_array_equal(y, [1, 0, 1, 0, 1, 0])

    def test_fit_only_alternative_positives_are_skipped_identically_for_both_models(self):
        a = row(0, correct=0)
        b = copy.deepcopy(a)
        b.update(row_id=1, source_row_id=1, target=b['candidates'][1]['smiles'], top1_correct=False)
        for i, candidate in enumerate(b['candidates']):
            candidate['correct'] = i == 1
        matrices, _ = pilot.prepare([a, b])
        for family in ('learned_global', pilot.PRIMARY):
            _, _, weights, info = pilot.pairwise_data([a, b], matrices, family)
            self.assertEqual(info['contradictory_negative_pairs_skipped'], 2)
            self.assertEqual(info['candidate_pairs'], 2)
            self.assertEqual(info['fit_inputs_with_multiple_targets'], 1)
            self.assertEqual(sum(weights), 2)
        # An alternate target not present in the supplied FIT rows is not consulted.
        _, _, _, info = pilot.pairwise_data([a], [matrices[0]], pilot.PRIMARY)
        self.assertEqual(info['contradictory_negative_pairs_skipped'], 0)

    def test_tuning_prefers_frequency_when_rescues_and_harms_cancel(self):
        rows = [row(0, split='val', correct=1), row(1, split='val', correct=0)]
        matrices, _ = pilot.prepare(rows)
        fitted, grid = pilot.tune_method(rows, matrices, 'chemistry_rule', {})
        self.assertEqual(fitted['config']['alpha'], 0)
        self.assertEqual(fitted['validation']['changed'], 0)
        self.assertTrue(any(item['summary']['harmed'] == 1 and item['summary']['corrected'] == 1 for item in grid))

    def test_stable_ties_margin_and_model_off(self):
        record = row()
        matrix, _ = pilot.features(record)
        for scores, margin, expected in (([0, 0, -1], 0, 0), ([0, 2, 2], 0, 1), ([0, 2, 2], 2, 0)):
            self.assertEqual(pilot.choose([record], [matrix], [np.array(scores)],
                                         {'alpha': 1., 'margin': margin}, pilot.PRIMARY), [expected])
        self.assertEqual(pilot.choose([record], [matrix], [np.array([0, 2, 2])],
                                     {'alpha': 0., 'margin': 0.}, pilot.PRIMARY), [0])

    def test_all_rows_accounting_includes_rescues_harms_and_no_candidates(self):
        rows = [row(0, correct=1), row(1, correct=0), row(2, correct=0),
                row(3, correct=None), row(4, correct=None)]
        rows[-1]['candidates'] = []
        outcomes = pilot.outcome_rows(rows, [1, 1, 0, 1, None], 10, [True]*4+[False])
        summary = pilot.summarize(outcomes)
        for key, value in {'rows': 5, 'corrected': 1, 'harmed': 1, 'unchanged_correct': 1,
                           'unchanged_wrong': 2, 'changed_wrong_to_wrong': 1, 'net_correct': 0,
                           'target_absent_from_top_k': 2, 'covered_rows': 4}.items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(summary['accuracy'], .4)

    def test_bootstrap_resamples_groups_not_individual_rows(self):
        outcomes = [{'split_group': 'large', 'baseline_correct': False, 'correct': True} for _ in range(100)]
        outcomes.append({'split_group': 'small', 'baseline_correct': True, 'correct': False})
        result = pilot.cluster_bootstrap(outcomes)
        self.assertEqual(result['resamples'], 10000)
        self.assertEqual((result['lower'], result['upper']), (-1., 1.))
        self.assertAlmostEqual(result['point_estimate'], 99/101)
        self.assertEqual(result, pilot.cluster_bootstrap(list(reversed(outcomes))))

    def test_insufficient_training_signal_has_no_fitted_model(self):
        models, info = pilot.fit_models([row(0)], pilot.prepare([row(0)])[0], pilot.PRIMARY)
        self.assertEqual(models, {})
        self.assertEqual(info['status'], 'insufficient_pairs_frequency_fallback')


class LeakageTests(unittest.TestCase):
    def test_official_split_and_namespace_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'source_split=train'):
            pilot.validate_records([row(split='test')], 'train')
        bad = row()
        bad['sequence_id'] = '7'
        with self.assertRaisesRegex(ValueError, 'namespaced'):
            pilot.validate_records([bad], 'train')
        bad['sequence_id'] = 'PM'
        pilot.validate_records([bad], 'train')

    def test_actual_canonical_input_sequence_and_group_overlap_are_rejected(self):
        fit = row(0, reactant='CCO')
        audit = row(1, split='val', reactant='OCC')
        with self.assertRaisesRegex(ValueError, 'reactants'):
            pilot.assert_disjoint(pilot.signature([fit]), pilot.signature([audit]))
        audit = row(1, split='val')
        audit['split_group'] = fit['split_group']
        with self.assertRaisesRegex(ValueError, 'groups'):
            pilot.assert_disjoint(pilot.signature([fit]), pilot.signature([audit]))
        a, b = row(2, split='val'), row(3, split='val')
        b['sequence_id'] = a['sequence_id']
        with self.assertRaisesRegex(ValueError, 'sequences'):
            pilot.assert_disjoint(pilot.signature([a]), pilot.signature([b]))
        a['sequence_id'] = b['sequence_id'] = 'RS'
        pilot.assert_disjoint(pilot.signature([a]), pilot.signature([b]))

    def test_related_rows_cannot_have_different_bootstrap_groups(self):
        a, b = row(0), row(1)
        b['reactant'] = a['reactant']
        with self.assertRaisesRegex(ValueError, 'inconsistent split_group'):
            pilot.validate_records([a, b], 'train')
        b['split_group'] = a['split_group']
        pilot.validate_records([a, b], 'train')

    def test_manifest_prevents_development_overlap_and_membership_mutation(self):
        fit, audit = row(0), row(5, split='val')
        manifest = manifest_for([audit])
        pilot.manifest_disjoint(pilot.signature([fit]), manifest)
        pilot.manifest_membership([audit], manifest)
        collision = copy.deepcopy(manifest)
        collision[0]['split_group'] = fit['split_group']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            pilot.manifest_disjoint(pilot.signature([fit]), collision)
        changed = copy.deepcopy(audit)
        changed['source_line_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source_line_sha256'):
            pilot.manifest_membership([changed], manifest)
        changed = copy.deepcopy(audit)
        changed['selection_cohort'] = 'outcome selected'
        with self.assertRaisesRegex(ValueError, 'selection_cohort'):
            pilot.manifest_membership([changed], manifest)

    def test_writer_exact_labels_keep_raw_canonical_duplicates_distinct(self):
        record = row(smiles=['CCO', 'OCC'], reactant='CCO', counts=[32, 30])
        pilot.validate_records([record], 'train')
        record['candidates'][0]['correct'] = True
        with self.assertRaisesRegex(ValueError, 'exact saved-writer'):
            pilot.validate_records([record], 'train')


class FrozenWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fit, cls.tune, cls.audit = (cls.root/(name+'.jsonl') for name in ('fit', 'tune', 'audit'))
        fit = [row(i) for i in range(12)]
        tune = [row(i, split='val') for i in range(15, 21)]
        tune[0]['selection_cohort'] = 'motif_enriched'
        audit = [row(i, split='val', correct=i % 2) for i in range(25, 31)]
        audit.append(row(32, split='val', correct=None))
        audit[-1]['candidates'] = []
        audit.append(row(33, split='val', correct=None))
        audit[-1]['candidates'][0].update(smiles='OPAQUE[broken', parseable=False)
        audit[-1]['rerank_eligible'] = False
        cls.audit_rows = audit
        for path, rows in ((cls.fit, fit), (cls.tune, tune), (cls.audit, audit)):
            write_rows(path, rows)
        cls.manifest = cls.root/'audit-metadata.json'
        cls.manifest.write_text(json.dumps(manifest_for(audit)))
        cls.fit_args = SimpleNamespace(fit=cls.fit, tune=cls.tune, out=cls.root/'model',
                                       audit_manifest=cls.manifest, audit_split='val')
        with redirect_stdout(io.StringIO()):
            cls.receipt = pilot.fit_tune(cls.fit_args)
        cls.eval_args = SimpleNamespace(records=cls.audit, model=cls.root/'model/frozen.joblib',
                                        out=cls.root/'assessment')
        with redirect_stdout(io.StringIO()):
            cls.report = pilot.evaluate(cls.eval_args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_four_methods_primary_targeted_and_separate_strata(self):
        self.assertEqual(set(self.report['methods']), set(pilot.METHODS))
        self.assertEqual(self.report['primary_method'], 'learned_targeted')
        self.assertEqual(self.report['status'], 'group_disjoint_official_validation_audit')
        self.assertEqual(set(self.receipt['validation_strata']['frequency']), {'random', 'motif_enriched'})
        for method in pilot.METHODS:
            result = self.report['methods'][method]
            self.assertEqual(result['rows'], 8)
            self.assertEqual(result['no_saved_candidates'], 1)
            self.assertEqual(result['unsupported_fallback_rows'], 1)
            self.assertEqual(set(result['selection_strata']), {'random'})
        self.assertFalse(self.report['primary_evidence']['automatic_benchmark_resumption'])
        self.assertEqual(self.receipt['protocol']['k'], 10)
        self.assertIn(self.receipt['selected_configurations'][pilot.PRIMARY]['alpha'], (.25, .5))
        self.assertEqual(self.report['methods'][pilot.PRIMARY]['corrected'], 3)
        self.assertEqual(self.report['methods'][pilot.PRIMARY]['harmed'], 3)
        self.assertEqual(self.report['targeted_minus_global_ablation']['rows'], 8)

    def test_frozen_fit_only_contains_train_and_tune_only_val(self):
        for role, split in (('fit', 'train'), ('tune', 'val')):
            self.assertEqual({r['source_split'] for r in self.receipt['development_rows'][role]}, {split})
        self.assertEqual(self.receipt['protocol']['primary_method'], pilot.PRIMARY)
        self.assertEqual(self.receipt['inputs']['audit_manifest_sha256'], pilot.file_hash(self.manifest))

    def test_repeat_assessment_verifies_same_artifacts(self):
        before = pilot.file_hash(self.eval_args.out/'decisions.jsonl')
        with redirect_stdout(io.StringIO()):
            again = pilot.evaluate(self.eval_args)
        self.assertEqual(again, self.report)
        self.assertEqual(before, pilot.file_hash(self.eval_args.out/'decisions.jsonl'))

    def test_saved_artifact_tampering_is_rejected(self):
        path = self.eval_args.out/'decisions.jsonl'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'artifacts changed'):
                pilot.evaluate(self.eval_args)
        finally:
            path.write_bytes(original)

    def test_changed_output_binding_is_rejected(self):
        changed = copy.copy(self.eval_args)
        changed.out = self.root/'different-assessment'
        with self.assertRaisesRegex(ValueError, 'already bound'):
            pilot.evaluate(changed)

    def test_changed_records_even_same_membership_are_rejected(self):
        path = self.root/'mutated-audit.jsonl'
        changed = copy.deepcopy(self.audit_rows)
        changed[0]['candidate_entropy'] = 99
        write_rows(path, changed)
        args = copy.copy(self.eval_args)
        args.records = path
        with self.assertRaisesRegex(ValueError, 'already bound'):
            pilot.evaluate(args)

    def test_modified_runtime_and_manifest_membership_fail_before_scoring(self):
        with patch.object(pilot, 'versions', return_value={'rdkit': 'different'}):
            with self.assertRaisesRegex(ValueError, 'implementation/runtime'):
                pilot.evaluate(self.eval_args)
        records = self.root/'wrong-membership.jsonl'
        write_rows(records, self.audit_rows[:-1])
        args = copy.copy(self.eval_args)
        args.records = records
        with self.assertRaisesRegex(ValueError, 'membership'):
            pilot.evaluate(args)

    def test_test_assessment_requires_frozen_manifest_before_training(self):
        args = copy.copy(self.fit_args)
        args.audit_split, args.audit_manifest = 'test', None
        with self.assertRaisesRegex(ValueError, 'precommitted metadata'):
            pilot.fit_tune(args)


if __name__ == '__main__':
    unittest.main()
