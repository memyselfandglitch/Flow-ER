import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import diagnose_rank_failures as diagnosis


class FailureDiagnosisTests(unittest.TestCase):
    def row(self,truth,metrics,candidates):
        source=truth+'|one\n'
        reactant=diagnosis.reference(truth.split('>>')[0])
        meta={'reactant_key':hashlib.sha256(reactant.encode()).hexdigest(),
              'source_line_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'sequence_id':'test:one','split_group':'group'}
        return diagnosis.analyze_saved(source,f'{metrics}|0|{candidates}\n',0,meta)

    def test_unsupported_candidate_is_retained_and_abstains(self):
        target=diagnosis.reference('CCO')
        row=self.row('CCO>>CCO',[20,44,0,0,0],[('INVALID_SMILES',44,True),(target,20,True)])
        self.assertEqual(row['target_rank'],2)
        self.assertEqual(row['candidates'][0]['smiles'],'INVALID_SMILES')
        self.assertFalse(row['rerank_eligible'])
        self.assertEqual(row['unsupported_candidate_ranks'],[1])

    def test_absent_target_does_not_control_eligibility(self):
        alternative=diagnosis.reference('COC')
        row=self.row('CCO>>CCO',[0,64,0,0,0],[(alternative,64,True)])
        self.assertIsNone(row['target_rank'])
        self.assertTrue(row['rerank_eligible'])

    def test_stable_ties_preserve_frequency_baseline(self):
        target=diagnosis.reference('CCO');alternative=diagnosis.reference('COC')
        row=self.row('CCO>>CCO',[32,32,0,0,0],[(alternative,32,True),(target,32,True)])
        self.assertEqual(row['target_rank'],2)
        self.assertTrue(row['target_tied_for_first'])
        self.assertEqual(row['top1_target_count_gap'],0)

    def test_protonation_and_connectivity_signatures(self):
        acid=diagnosis.chemistry('CC(=O)O');anion=diagnosis.chemistry('CC(=O)[O-]')
        self.assertEqual(acid['heavy_bond_signature'],anion['heavy_bond_signature'])
        carbonyl=diagnosis.chemistry('C=O');alcohol=diagnosis.chemistry('CO')
        self.assertEqual(carbonyl['connectivity_signature'],alcohol['connectivity_signature'])
        self.assertNotEqual(carbonyl['heavy_bond_signature'],alcohol['heavy_bond_signature'])
        self.assertNotEqual(alcohol['connectivity_signature'],diagnosis.chemistry('C.O')['connectivity_signature'])

    def test_reference_tally_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError,'Alignment'):
            self.row('CCO>>CCO',[20,44,0,0,0],[(diagnosis.reference('COC'),64,True)])


if __name__=='__main__': unittest.main()
