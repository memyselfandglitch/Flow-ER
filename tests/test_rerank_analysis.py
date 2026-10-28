import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('rerank_analysis', Path(__file__).parents[1]/'scripts/rerank_analysis.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class RankingTests(unittest.TestCase):
    def test_strict_error_and_tie_are_separate(self):
        strict = analysis.analyze_row('CCO>>CCO|one', "[2, 5, 0, 1, 0]|0|[('COC', 5, True), ('CCO', 2, True)]", 0, 10)
        self.assertEqual(strict['target_rank'], 2)
        self.assertFalse(strict['target_tied_for_first'])
        tied = analysis.analyze_row('CCO>>CCO|one', "[4, 4, 0, 0, 0]|0|[('COC', 4, True), ('CCO', 4, True)]", 0, 10)
        self.assertFalse(tied['top1_correct'])
        self.assertTrue(tied['target_tied_for_first'])
        self.assertEqual((tied['target_rank_best_tie'], tied['target_rank_worst_tie']), (1, 2))

    def test_maps_and_equivalent_smiles_merge(self):
        row = analysis.analyze_row('[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]|one', "[6, 2, 0, 0, 0]|0|[('CCO', 3, True), ('OCC', 3, True), ('COC', 2, True)]", 0, 10)
        self.assertEqual(row['candidates'][0]['count'], 6)
        self.assertTrue(row['top1_correct'])
        self.assertTrue(row['candidates'][0]['no_op'])

    def test_alignment_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'alignment'):
            analysis.analyze_row('CCO>>COC|one', "[6, 2, 0, 0, 0]|0|[('CCO', 6, True), ('COC', 2, True)]", 0, 10)

    def test_absent_target_is_not_rescuable(self):
        row = analysis.analyze_row('CCO>>CCO|one', "[0, 8, 0, 0, 0]|0|[('COC', 8, True)]", 0, 10)
        self.assertIsNone(row['target_rank'])
        self.assertFalse(row['top1_correct'])

    def test_pathway_and_reactant_transitive_grouping(self):
        rows = [{'sequence_id':'1','reactant':'A'}, {'sequence_id':'1','reactant':'B'},
                {'sequence_id':'2','reactant':'B'}, {'sequence_id':'2','reactant':'C'},
                {'sequence_id':'PM','reactant':'D'}, {'sequence_id':'PM','reactant':'E'}]
        groups = analysis.components(rows)
        self.assertEqual(len(set(groups[:4])), 1)
        self.assertEqual(len(set(groups)), 3)

    def test_skeleton_keeps_connectivity_but_ignores_protonation(self):
        self.assertEqual(analysis.heavy_skeleton('CC(=O)O'), analysis.heavy_skeleton('CC(=O)[O-]'))
        self.assertNotEqual(analysis.heavy_skeleton('CCO'), analysis.heavy_skeleton('COC'))


if __name__ == '__main__':unittest.main()
