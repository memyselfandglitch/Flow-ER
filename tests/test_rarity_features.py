import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rarity_features import signatures


class RarityFeatureTests(unittest.TestCase):
    def test_atom_maps_and_fragment_order_do_not_define_rarity(self):
        first = signatures('[CH3:1][OH:2].[Cl-:3]', '[CH3:1][O-:2].[Cl-:3]')
        other = signatures('[Cl-:17].[OH:92][CH3:48]', '[Cl-:17].[O-:92][CH3:48]')
        self.assertEqual(first, other)

    def test_unchanged_spectators_cancel(self):
        self.assertEqual(signatures('CCO', 'CC[O-]'), signatures('CCO.CCCC', 'CC[O-].CCCC'))

    def test_direction_of_change_is_retained(self):
        self.assertNotEqual(signatures('CCO', 'CC[O-]'), signatures('CC[O-]', 'CCO'))

    def test_identity_has_separate_signature(self):
        self.assertEqual(signatures('CCO', 'OCC'), ('no_local_change', 'no_local_change'))


if __name__ == '__main__':
    unittest.main()
