"""Whole-molecule recognition and label-independent assignment tests."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_solvent_review as review


class SolventReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dictionary, cls.catalog = review.load_dictionary()

    def recognize(self, source, candidates):
        return review.recognize(source, candidates, self.catalog)

    def test_canonical_matching_ignores_map_numbers_and_ordinary_explicit_h_not_charge(self):
        self.assertEqual(review.whole_molecule_key('[H][O:5][C:3]([H])([H])[H]'), review.whole_molecule_key('CO'))
        self.assertNotEqual(review.whole_molecule_key('C[O-]'), review.whole_molecule_key('CO'))
        self.assertNotEqual(review.whole_molecule_key('[2H]O'), review.whole_molecule_key('O'))

    def test_substructure_is_not_a_match(self):
        result = self.recognize('OCCc1ccccc1', ['OCCc1ccccc1'])
        self.assertEqual(result['recognized_molecules'], [])
        result = self.recognize('CC1CCCO1', ['CC1CCCO1'])
        self.assertEqual([m['name'] for m in result['recognized_molecules']], ['2-methyltetrahydrofuran'])

    def test_sole_unchanged_organic_solvent_is_assumed_not_documented(self):
        r = self.recognize('CN.Cl.C1CCOC1', ['C[NH3+].[Cl-].C1CCOC1', 'CN.Cl.C1CCOC1'])
        self.assertEqual(r['status'], 'assumed'); self.assertEqual(r['proposed_nominal_solvent'], 'THF')
        self.assertIsNone(r['experimental_solvent_documentation'])

    def test_water_and_alcohols_trigger_possible_mixture_review(self):
        for co in ('O', 'CO', 'CCO', 'CC(O)C'):
            source = 'CN.Cl.C1CCOC1.'+co
            r = self.recognize(source, [source])
            self.assertEqual(r['status'], 'unresolved'); self.assertTrue(r['review_required'])
            self.assertIsNone(r['proposed_nominal_solvent'])
            self.assertIn('possible_mixed_solvent_or_reagent_roles', r['review_flags'])

    def test_multiplicity_never_establishes_dominant_solvent(self):
        source = '.'.join(['C1CCOC1']*10+['O'])
        r = self.recognize(source, [source]); self.assertEqual(r['status'], 'unresolved')
        self.assertIsNone(r['proposed_nominal_solvent'])

    def test_all_retained_candidates_must_preserve_recognized_molecule(self):
        source = 'CN.Cl.C1CCOC1'
        r = self.recognize(source, [source]*9+['CN.[Cl-].C1CC[OH+]C1'])
        self.assertEqual(r['status'], 'unresolved')
        self.assertEqual(r['recognized_molecules'][0]['changed_or_unreadable_candidate_ranks'], [10])

    def test_no_silent_water_or_gas_assignment(self):
        for source, candidates in [('CN.Cl.O', ['CN.Cl.O']), ('CN.Cl', ['CN.Cl']), ('C1CCOC1', [])]:
            r = self.recognize(source, candidates)
            self.assertEqual(r['status'], 'unresolved'); self.assertIsNone(r['proposed_nominal_solvent'])

    def test_nominal_assignment_can_need_separate_alpb_mapping_review(self):
        r = self.recognize('CC(O)C', ['CC(O)C'])
        self.assertEqual(r['status'], 'assumed'); self.assertIsNone(r['proposed_alpb_name'])
        self.assertTrue(r['review_required'])

    def test_decision_is_independent_of_reference_frequency_correctness_and_energy_fields(self):
        row = {'source_row_id': 9, 'source_line_sha256': 'immutable', 'target': 'reference-A',
            'target_rank': 2, 'top1_correct': False,
            'candidates': [{'smiles': 'CN.Cl.C1CCOC1', 'count': 60, 'rank': 1, 'correct': False},
                           {'smiles': 'C[NH3+].[Cl-].C1CCOC1', 'count': 4, 'rank': 2, 'correct': True}]}
        changed = copy.deepcopy(row); changed.update(target='other', target_rank=1, top1_correct=True)
        for c in changed['candidates']: c.update(count=1, correct=False, energy=-999.)
        a = review.recognize_saved_record(row, 'CN.Cl.C1CCOC1>>ignored-target|9', self.catalog)
        b = review.recognize_saved_record(changed, 'CN.Cl.C1CCOC1>>different-ignored-target|9', self.catalog)
        # Hashes deliberately detect saved-pool mutations; decisions never use these fields.
        a.pop('candidate_pool_sha256'); b.pop('candidate_pool_sha256')
        self.assertEqual(a, b)

    def test_fragment_position_does_not_change_decision(self):
        a = self.recognize('C1CCOC1.CN.Cl', ['C1CCOC1.C[NH3+].[Cl-]'])
        b = self.recognize('Cl.CN.C1CCOC1', ['[Cl-].C[NH3+].C1CCOC1'])
        self.assertEqual(a, b)


if __name__ == '__main__': unittest.main()
