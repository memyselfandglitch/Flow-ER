import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('saved_ranks',Path(__file__).parents[1]/'scripts/summarize_saved_ranks.py')
ranker=importlib.util.module_from_spec(spec);spec.loader.exec_module(ranker)


class SavedRankTests(unittest.TestCase):
    def audit(self,metrics,candidates):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'gt').write_text('CCO>>CCO|one\n')
            (root/'pred').write_text(f'{metrics}|0|{candidates}\n')
            return ranker.summarize(root/'gt',root/'pred')

    def test_retains_unparseable_saved_competitor_in_ranking(self):
        target=ranker.target_string('CCO')
        result=self.audit([20,44,0,0,0],[('not_a_parseable_smiles',44,True),(target,20,True)])
        self.assertEqual(result['counts']['top1_wrong'],1)
        self.assertEqual(result['counts']['top3_correct_top1_wrong'],1)

    def test_ties_use_saved_order_and_are_reported(self):
        target=ranker.target_string('CCO')
        result=self.audit([32,32,0,0,0],[(ranker.target_string('COC'),32,True),(target,32,True)])
        self.assertEqual(result['counts']['top1_wrong_but_target_tied_for_first'],1)
        self.assertEqual(result['counts']['top3_strict_frequency_errors'],0)

    def test_empty_candidates_remain_in_denominator(self):
        result=self.audit([0,0,0,64,0],[])
        self.assertEqual(result['counts']['rows'],1)
        self.assertEqual(result['counts']['top5_miss'],1)

    def test_reference_tally_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError,'Target/tally'):
            self.audit([12,52,0,0,0],[(ranker.target_string('COC'),64,True)])


if __name__=='__main__': unittest.main()
