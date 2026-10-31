"""Checks that label diagnostics do not conflate proton movement and resonance."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from audit_recorded_targets import compare_states, fragments, hydrogen_connectivity, input_functions, mapped_change


class LabelAuditTests(unittest.TestCase):
    def test_proton_transfer_is_not_only_resonance(self):
        result = compare_states('CC(=O)O.CN', 'CC(=O)[O-].C[NH3+]')
        self.assertEqual(result['category'], 'same_heavy_bonds_H_charge_difference')
        self.assertFalse(result['same_H_attachment_connectivity'])

    def test_resonance_keeps_hydrogen_attachment(self):
        a, b = 'CN(C)C=O', 'C[N+](C)=C[O-]'
        self.assertEqual(hydrogen_connectivity(a), hydrogen_connectivity(b))
        self.assertEqual(compare_states(a, b)['category'], 'rdkit_resonance_related')

    def test_largest_fragment_proxy_does_not_erase_other_fragments(self):
        a, b = fragments('CCCO.O'), fragments('CCCO.[OH-]')
        self.assertEqual(a['largest'], b['largest'])
        self.assertNotEqual(a['all'], b['all'])
        self.assertIsNone(fragments('CCO.CCN')['largest'])

    def test_serialization_is_not_a_new_chemical_branch(self):
        self.assertEqual(compare_states('CCO', 'OCC')['category'], 'serialization_only')

    def test_connectivity_change_is_distinguished(self):
        self.assertEqual(compare_states('CCO', 'COC')['category'], 'different_heavy_connectivity')

    def test_opaque_saved_candidate_is_explicit(self):
        self.assertEqual(compare_states('OPAQUE[broken', 'CCO')['category'], 'unsupported_saved_string')
        self.assertIsNone(fragments('OPAQUE[broken'))

    def test_actual_input_helpers_preserve_atom_index_order(self):
        token, matrix = input_functions(Path(__file__).parents[1]/'utils/data_utils.py')
        a, b = '[O:1]([H:2])[H:3]', '[H:3][O:1][H:2]'
        np.testing.assert_array_equal(token(a)[0], token(b)[0])
        np.testing.assert_array_equal(matrix(a), matrix(b))

    def test_mapped_proton_motion_is_visible(self):
        a = '[O-:1].[N:2]([H:3])([H:4])[H:5]'
        b = '[O:1][H:3].[N-:2]([H:4])[H:5]'
        result = mapped_change(a, b)
        self.assertEqual(result['heavy_bond_changes'], [])
        self.assertEqual(len(result['hydrogen_bond_changes']), 2)
        self.assertEqual(result['changed_heavy_H_counts'], [1, 2])


if __name__ == '__main__':
    unittest.main()
