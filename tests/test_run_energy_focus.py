"""Focused execution integration tests; mocked scores, no physical work."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_energy_focus import assess,physical_records
from xtb_671_reference_state import translated_arguments


def case(i):
    return {'source_row_id':i,'reactant':'INPUT','target':'A','target_rank':2,
        'solvent':{'nominal_name':'THF'},'recorded_alternatives':[{'smiles':'A'}],
        'candidates':[{'rank':1,'smiles':'B','count':63,'correct':False},
                      {'rank':2,'smiles':'A','count':1,'correct':True}]}


def score(i,rank,energy,status='ok'):
    return {'row_id':i,'rank':rank,'smiles':'B' if rank==1 else 'A',
        'status':status,'energy_hartree':energy,'computation_protocol_sha256':'frozen-one',
        'first_conformer_energy_hartree':energy,'conformer_spread_kcal_mol':0.,
        'fragment_provenance':[]}


class FocusTests(unittest.TestCase):
    def test_backend_spelling_bridge_changes_only_alpb_state(self):
        arguments=['bar1mol','--alpb','thf','bar1mol','--etemp','300']
        self.assertEqual(translated_arguments(arguments),
                         ['bar1mol','--alpb','thf','bar1M','--etemp','300'])
        self.assertEqual(translated_arguments(['--alpb','thf','reference']),
                         ['--alpb','thf','reference'])

    def test_physical_scorer_does_not_receive_target_correctness_or_frequency(self):
        rows=physical_records([case(1)])
        self.assertEqual(set(rows[0]),{'row_id','reactant','candidates'})
        self.assertTrue(all(set(c)=={'rank','smiles'} for c in rows[0]['candidates']))

    def test_failed_row_retained_in_full_denominator_and_never_partially_ranked(self):
        rows=[case(1),case(2)]
        values=[score(1,1,-1.),score(1,2,-2.),score(2,1,None,'failed'),score(2,2,-10.)]
        result,summary=assess(rows,values)
        self.assertEqual(summary['complete_pool_rows'],1)
        self.assertEqual(summary['references_promoted_to_energy_rank1'],1)
        self.assertEqual(summary['recovery_all_selected'],.5)
        self.assertEqual(summary['recovery_fully_scored'],1.)
        self.assertEqual(result[2]['status'],'unresolved')
        self.assertIsNone(result[2]['reference_rank_after'])

    def test_no_complete_pool_has_undefined_conditional_recovery(self):
        _,s=assess([case(1)],[score(1,1,None,'failed'),score(1,2,-1.)])
        self.assertIsNone(s['recovery_fully_scored'])
        self.assertEqual(s['recovery_all_selected'],0.)

    def test_incomplete_or_duplicate_candidate_accounting_rejected(self):
        for values in ([score(1,1,-1.)],[score(1,1,-1.),score(1,1,-2.)]):
            with self.assertRaises(ValueError):assess([case(1)],values)


if __name__=='__main__':unittest.main()
