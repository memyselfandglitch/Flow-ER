"""Boundary, identity and paired-statistic tests; no model training."""
from pathlib import Path
import sys
import unittest
import tempfile
import subprocess
import json
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase0_reranking_audit import (AuditReader, ROOT, input_functions, model_input_hash,
    label_diagnostics, lookup_predictions, assign_groups, fold_lookup, protocol)
from phase0_group_power import bootstrap_ratios, simulate_differences, sizes_for_n
from phased_reranking_contracts import residual_choice,soft_targets,bounded_failure_weight,correspondence_status


def row(i,x,target,winner,pool=None):
    pool=pool or [winner]
    return {'source_row_id':i,'exact_input':x,'normalized_input':x,'target':target,
            'target_rank':next((j+1 for j,y in enumerate(pool) if y==target),None),
            'candidates':[{'smiles':s,'count':64-j} for j,s in enumerate(pool)]}


class IdentityAndScopeTests(unittest.TestCase):
    def test_target_files_are_rejected_before_open(self):
        reader=AuditReader([ROOT/'results/reranking/pilot-v2/test-input-index.jsonl'])
        for relative in ['data/flower_new_dataset/test.txt',
                'results/reranking/full-validation-test20k-20261029/inference/inputs/test000.txt']:
            with self.assertRaises(PermissionError): reader.json(ROOT/relative)

    def test_exact_arrays_distinguish_atom_index_permutation(self):
        token,matrix=input_functions(ROOT/'utils/data_utils.py')
        a=model_input_hash('[CH3:1][OH:2]',token,matrix)
        self.assertEqual(a,model_input_hash('[OH:2][CH3:1]',token,matrix))
        self.assertNotEqual(a,model_input_hash('[CH3:2][OH:1]',token,matrix))
        # Element tokens/BE inputs do not represent isotope identity.
        self.assertEqual(a,model_input_hash('[13CH3:1][OH:2]',token,matrix))

    def test_transitive_input_and_network_grouping(self):
        rows=[row(i,str(i),'A','A') for i in range(4)]
        rows[1]['normalized_input']=rows[0]['normalized_input']
        rows[2]['exact_input']=rows[1]['exact_input']
        metadata={i:{'split_group':'network' if i in (2,3) else str(i)} for i in range(4)}
        assign_groups(rows,metadata)
        self.assertEqual(len({r['phase0_group'] for r in rows}),1)
        self.assertEqual(len({r['fold'] for r in rows}),1)

    def test_execution_holds_are_declared(self):
        p=protocol()
        self.assertTrue(all(p['execution_holds'].values()))
        self.assertEqual(p['minimum_worthwhile_gain_pp'],1.)
        self.assertEqual(p['primary_inference']['bootstrap_replicates'],9999)

    def test_later_phase_execution_is_blocked_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'protocol.json').write_text(json.dumps(protocol()))
            before=(p/'protocol.json').read_bytes()
            for phase in (1,2,3):
                run=subprocess.run([sys.executable,str(ROOT/'scripts/phased_reranking_gate.py'),
                    '--experiment',d,'--phase',str(phase),'--execute'],capture_output=True,text=True)
                self.assertNotEqual(run.returncode,0)
                self.assertIn('explicitly holds execution',run.stderr)
            self.assertEqual((p/'protocol.json').read_bytes(),before)
            self.assertEqual([f.name for f in p.iterdir()],['protocol.json'])


class LabelDiagnosticTests(unittest.TestCase):
    def test_empirical_ceiling_and_minority_winners(self):
        rows=[row(0,'x','A','B',['B','A']),row(1,'x','A','B',['B','A']),
              row(2,'x','B','B',['B','A']),row(3,'y','C','C')]
        result=label_diagnostics(rows,'exact_input')
        self.assertEqual(result['best_fixed_choice_matches'],3)
        self.assertEqual(result['best_fixed_choice_accuracy'],.75)
        self.assertEqual(result['strict_top1'],2)
        self.assertEqual(result['minority_recorded_winner'],3)
        self.assertEqual(result['minority_winner_row_correct'],1)
        self.assertEqual(result['any_recorded_top1'],4)

    def test_comajority_is_not_minority(self):
        rows=[row(0,'x','A','B',['B','A']),row(1,'x','B','A',['A','B'])]
        d=label_diagnostics(rows,'exact_input')
        self.assertEqual(d['minority_recorded_winner'],0)
        self.assertEqual(d['mode_tied_inputs'],1)
        self.assertEqual(d['best_fixed_choice_matches'],1)

    def test_lookup_uses_fitting_labels_only_and_retains_fallback(self):
        fitting=[row(0,'x','A','A'),row(1,'x','A','A'),row(2,'x','B','B')]
        assessed=[row(3,'x','B','B',['B','A']),row(4,'unseen','A','B',['B','A'])]
        a=lookup_predictions(fitting,assessed,'exact_input')
        assessed[0]['target']='UNSEEN_TARGET'
        b=lookup_predictions(fitting,assessed,'exact_input')
        self.assertEqual([r['prediction'] for r in a],[r['prediction'] for r in b])
        self.assertEqual(a[0]['prediction'],'A');self.assertTrue(a[0]['used_lookup'])
        self.assertFalse(a[1]['covered']);self.assertTrue(a[1]['fallback'])
        self.assertEqual(a[1]['prediction'],'B')

    def test_missing_majority_cannot_insert_candidate(self):
        a=lookup_predictions([row(0,'x','A','A')],[row(1,'x','B','B')],'exact_input')[0]
        self.assertTrue(a['covered']);self.assertTrue(a['mode_unavailable'])
        self.assertEqual(a['prediction'],'B')

    def test_fold_leakage_is_rejected(self):
        rows=[dict(row(i,'x' if i<2 else str(i),'A','A'),fold=i,phase0_group=str(i)) for i in range(5)]
        with self.assertRaisesRegex(ValueError,'Fold leakage'): fold_lookup(rows,'exact_input')


class PairedPowerTests(unittest.TestCase):
    def test_zero_intervention_has_zero_interval(self):
        values=bootstrap_ratios([1,5,3],[0,0,0],np.random.default_rng(1),100)
        np.testing.assert_array_equal(values,np.zeros(100))

    def test_bootstrap_is_row_weighted_not_mean_of_group_rates(self):
        class FixedDraws:
            def multinomial(self,n,p,size):
                self.last=(n,p,size)
                return np.array([[1,1],[2,0],[0,2]])
        rng=FixedDraws()
        values=bootstrap_ratios([1,9],[1,0],rng,3)
        np.testing.assert_allclose(values,[.1,1.,0.])
        self.assertEqual(rng.last[0],2)

    def test_compression_counts_duplicate_groups(self):
        class FixedDraws:
            def multinomial(self,n,p,size):
                self.n=n;self.p=p
                return np.array([[2,1]])
        rng=FixedDraws();values=bootstrap_ratios([1,1,3],[1,1,-1],rng,1)
        self.assertEqual(rng.n,3);np.testing.assert_allclose(rng.p,[2/3,1/3])
        np.testing.assert_allclose(values,[.2])

    def test_simulated_pairs_obey_counts_and_expected_effect(self):
        sizes=np.ones(200000,dtype=int)
        d,r,h=simulate_differences(sizes,.01,.04,.3,np.random.default_rng(14))
        self.assertEqual(d.sum(),r-h)
        self.assertTrue(np.isin(d,[-1,0,1]).all())
        self.assertAlmostEqual(d.mean(),.01,delta=.0015)

    def test_group_profile_preserves_exact_20k_sizes(self):
        n=sizes_for_n({'1':2,'3':2},8,1)
        self.assertEqual(n.tolist(),[1,1,3,3])

    def test_invalid_effect_discordance_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_differences(np.array([3]),.05,.01,.1,np.random.default_rng(1))


class ResidualContractTests(unittest.TestCase):
    def test_zero_policy_and_missing_scores_preserve_frequency(self):
        self.assertEqual(residual_choice([40,24],[0,100],0,0),0)
        self.assertEqual(residual_choice([40,24],None,1,0),0)
        self.assertEqual(residual_choice([40,24],[0,np.nan],1,0),0)
        self.assertIsNone(residual_choice([],[],1,0))

    def test_additive_log_prior_and_strict_margin(self):
        self.assertEqual(residual_choice([32,16],[0,1],1,.2),1)
        self.assertEqual(residual_choice([32,16],[0,1],1,.4),0)
        self.assertEqual(residual_choice([32,32],[0,0],1,0),0)
        with self.assertRaises(ValueError): residual_choice([64,64],[0,0],1,0)

    def test_other_recorded_outcomes_remain_positive(self):
        q,mass=soft_targets(['B','A','unknown'],{'A':3,'B':2,'C':5})
        np.testing.assert_allclose(q,[.4,.6,0]);self.assertEqual(mass,.5)
        self.assertEqual(bounded_failure_weight(q,True),1.)

    def test_missing_positive_is_skipped_not_mislabeled(self):
        q,mass=soft_targets(['B'],{'A':3})
        self.assertIsNone(q);self.assertEqual(mass,0.)
        self.assertEqual(bounded_failure_weight(q,True),0.)

    def test_failure_emphasis_is_bounded(self):
        self.assertEqual(bounded_failure_weight(np.array([0.,1.]),True),2.)
        self.assertEqual(bounded_failure_weight(np.array([0.,1.]),False),1.)

    def test_element_bijection_alone_does_not_prove_correspondence(self):
        self.assertEqual(correspondence_status([1,2],[1,2],['C','O'],['C','O'],False),'unverified_provenance')
        self.assertEqual(correspondence_status([1,2],[0,0],['C','O'],['C','O'],True),'maps_missing')
        self.assertEqual(correspondence_status([1,2],[2,1],['C','O'],['O','C'],True),'verified')
        self.assertEqual(correspondence_status([1,2],[2,1],['C','O'],['C','O'],True),'mapped_element_mismatch')
        self.assertEqual(correspondence_status([1,2],[1,2],['C'],['C'],True),'inventory_length_mismatch')


if __name__=='__main__': unittest.main()
