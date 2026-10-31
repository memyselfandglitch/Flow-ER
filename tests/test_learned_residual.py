"""Fit-level leakage, objective, immutable-pool and resumability checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy import sparse
import learned_residual_features as features
import learned_residual_model as model
import train_learned_residual as runner


def fixture(n=8):
    rows=[]
    for i in range(n):
        rows.append({'source_row_id':i,'normalized_input':f'input{i}','exact_input':f'exact{i}',
            'group':f'group{i}','fold':i%4,'target':'B' if i%3==0 else 'A',
            'reactant':'CO','sample_count':64,
            'candidates':[{'rank':1,'smiles':'A','count':40},{'rank':2,'smiles':'B','count':24}]})
    X=sparse.csr_matrix(np.array([[i%2,(i//2)%3,(i%5)-2] for i in range(2*n)],dtype=float))
    return rows,X,np.arange(0,2*n+1,2),[True]*n


class ObjectiveTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        rows,X,offsets,support=fixture()
        rows[1]['normalized_input']=rows[0]['normalized_input']
        data=model.fit_inputs(rows,X,offsets,support,list(range(8)),'bounded_failure')
        w=np.array([.2,-.1,.35]);eps=1e-6
        _,analytic=model.loss_gradient(w,data,.1)
        numeric=[]
        for i in range(len(w)):
            step=np.zeros_like(w);step[i]=eps
            numeric.append((model.loss_gradient(w+step,data,.1)[0]-model.loss_gradient(w-step,data,.1)[0])/(2*eps))
        np.testing.assert_allclose(analytic,numeric,rtol=1e-6,atol=1e-8)

    def test_alternative_positives_and_bounded_weights(self):
        rows,X,offsets,support=fixture(4)
        rows[1]['normalized_input']=rows[0]['normalized_input']
        rows[2]['target']='not_in_pool'
        data=model.fit_inputs(rows,X,offsets,support,list(range(4)),'bounded_failure')
        np.testing.assert_allclose(data['q'],[.5,.5,.5,.5,0,1])
        np.testing.assert_allclose(data['weights'],[.75,.75,1.5])
        self.assertEqual(data['info']['raw_emphasis_rows'],1)
        self.assertEqual(data['info']['skipped_loss_rows'],{'no_fitting_catalog_positive_in_pool':1})
        self.assertEqual(data['info']['scaling_candidates'],8)
        self.assertEqual(data['info']['loss_rows'],3)
        out=model.outcomes(rows,list(range(4)),[0]*4,support)
        self.assertEqual(model.summarize(out)['rows'],4)

    def test_all_fits_start_from_zero_and_nonconvergence_rejected(self):
        rows,X,offsets,support=fixture()
        data=model.fit_inputs(rows,X,offsets,support,list(range(8)),'uniform')
        def failed(fun,x0,**kwargs):
            np.testing.assert_array_equal(x0,np.zeros(3))
            return SimpleNamespace(success=False,fun=1.,x=np.ones(3),nit=1,message='iteration limit',jac=np.ones(3))
        with patch.object(model,'minimize',side_effect=failed):
            fitted=model.fit(data,.1)
        self.assertFalse(fitted['converged'])
        self.assertTrue(fitted['info']['initialized_from_zero'])
        ids=list(range(8));folds={i:i%2 for i in ids}
        selected,reports,rejected=model.cross_validated_policy(rows,X,offsets,support,ids,folds,2,
            lambda *args: {'converged':False})
        self.assertEqual(len(rejected),8)
        self.assertEqual(len(reports),1)
        self.assertEqual(selected['config']['lambda'],0)

    def test_budget_callback_prevents_returning_a_usable_fit(self):
        rows,X,offsets,support=fixture()
        data=model.fit_inputs(rows,X,offsets,support,list(range(8)),'uniform')
        def expired():raise runner.BudgetExpired('test limit')
        with self.assertRaises(runner.BudgetExpired):model.fit(data,.1,expired)


class LeakageTests(unittest.TestCase):
    def test_outer_labels_cannot_affect_fit_or_inner_selection(self):
        rows,X,offsets,support=fixture(10);ids=list(range(8))
        modified=deepcopy(rows)
        for row in modified[8:]:row['target']='held_out_reference_CHANGED'
        a=model.fit_inputs(rows,X,offsets,support,ids,'bounded_failure')
        b=model.fit_inputs(modified,X,offsets,support,ids,'bounded_failure')
        self.assertEqual(a['catalog'],b['catalog'])
        for k in ('q','scale','weights','prior'):np.testing.assert_array_equal(a[k],b[k])
        first=model.fit(a,.1);second=model.fit(b,.1)
        self.assertTrue(first['converged'])
        np.testing.assert_array_equal(first['weights'],second['weights'])
        folds={i:i%2 for i in ids}
        chosen_a=model.cross_validated_policy(rows,X,offsets,support,ids,folds,2)
        chosen_b=model.cross_validated_policy(modified,X,offsets,support,ids,folds,2)
        self.assertEqual(chosen_a,chosen_b)

    def test_scaler_does_not_see_held_out_features(self):
        rows,X,offsets,support=fixture();ids=list(range(6))
        changed=X.toarray();changed[12:]=1e8
        a=model.fit_inputs(rows,X,offsets,support,ids,'uniform')
        b=model.fit_inputs(rows,sparse.csr_matrix(changed),offsets,support,ids,'uniform')
        np.testing.assert_array_equal(a['scale'],b['scale'])
        np.testing.assert_array_equal(model.fit(a,.1)['weights'],model.fit(b,.1)['weights'])

    def test_group_or_either_input_identity_overlap_is_rejected(self):
        for key in ('group','normalized_input','exact_input'):
            rows,_,_,_=fixture(2);rows[1][key]=rows[0][key]
            with self.assertRaisesRegex(ValueError,'Fold leakage: '+key):model.assert_disjoint(rows,[0],[1])

    def test_read_allowlist_rejects_reserved_targets_before_opening(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('must not open')):
            for relative in ('data/flower_new_dataset/test.txt',
                'results/reranking/full-validation-test20k-20261029/inference/inputs/test000.txt'):
                with self.assertRaises(PermissionError):runner.read_allowed(runner.ROOT/relative)


class FeatureTests(unittest.TestCase):
    def reaction(self):
        return {'reactant':'CN.Cl','sample_count':64,'target':'[Cl-].C[NH3+]',
            'source_row_id':4,'network_id':'hidden','reference_maps':'not a feature','step_index':99,
            'candidates':[{'rank':1,'smiles':'CN.Cl','count':40,'features':['old']},
                          {'rank':2,'smiles':'C[NH3+].[Cl-]','count':24,'reference_terminal':True}]}

    def test_features_are_label_independent_and_pool_is_immutable(self):
        row=self.reaction();before=deepcopy(row)
        a,info=features.row_features(features.inference_view(row))
        self.assertTrue(info['supported'],info)
        self.assertEqual(a.shape,(2,len(features.FEATURE_NAMES)))
        self.assertGreater(a.nnz,0)
        self.assertEqual(row,before)
        modified=deepcopy(row);modified.update(target='DIFFERENT',step_index=0,network_id='other',reference_maps='other')
        modified['candidates'][0]['features']=['reference-derived-garbage']
        modified['candidates'][1]['reference_terminal']=False
        b,_=features.row_features(features.inference_view(modified))
        np.testing.assert_array_equal(a.toarray(),b.toarray())
        self.assertEqual(set(features.inference_view(row)),{'reactant','sample_count','candidates'})
        self.assertEqual(set(features.inference_view(row)['candidates'][0]),{'smiles','count'})

    def test_full_mixture_includes_multiple_solvent_molecules(self):
        d=features.describe('CN.O.C1CCOC1')
        self.assertTrue(d['supported'])
        self.assertEqual(d['properties']['fragments'],3)
        for name in ('water','THF'):self.assertEqual(d['solvents'][list(features.SOLVENTS).index(name)],1)
        # Recognition is whole molecule, not an ether-ring substructure assertion.
        d=features.describe('CN.O.CC1CCOC1')
        self.assertEqual(d['solvents'][list(features.SOLVENTS).index('THF')],0)

    def test_any_unsupported_candidate_falls_back_whole_pool(self):
        row=self.reaction();row['candidates'][1]['smiles']='NOT_SMILES'
        X,info=features.row_features(features.inference_view(row))
        self.assertFalse(info['supported']);self.assertEqual(X.nnz,0)
        self.assertEqual(model.choices([row],[0],[np.array([0,100])],[False],{'lambda':1,'margin':0}),[0])

    def test_counts_and_order_cannot_be_changed(self):
        row=self.reaction();row['candidates'].reverse()
        with self.assertRaisesRegex(ValueError,'original|Original'):features.row_features(features.inference_view(row))


class PolicyAndCheckpointTests(unittest.TestCase):
    def test_zero_policy_beats_harmful_overrides(self):
        rows,X,offsets,support=fixture(4)
        for row in rows:row['target']='A'
        scores={(l2,w):[np.array([0,100])]*4 for l2 in model.L2_GRID for w in model.WEIGHTINGS}
        chosen,_=model.choose_policy(rows,list(range(4)),scores,support)
        self.assertEqual(chosen['config']['lambda'],0)
        self.assertEqual(chosen['summary']['changed'],0)

    def test_correct_error_and_empty_pool_accounting(self):
        rows,_,_,support=fixture(5)
        rows[0]['target']='B';rows[1]['target']='A';rows[2]['target']='unavailable'
        rows[3]['candidates']=[];rows[4]['target']='A'
        support[4]=False
        counts=model.summarize(model.outcomes(rows,list(range(5)),[1,1,1,None,0],support))
        expected={'rows':5,'baseline_correct':2,'correct':2,'changed':3,'correction':1,
            'new_error':1,'wrong_to_wrong':1,'empty_pool':1,'fallback_unsupported':1}
        for k,v in expected.items():self.assertEqual(counts[k],v,k)

    def test_checkpoints_preserve_predictions_and_reject_mismatch(self):
        rows,X,offsets,support=fixture()
        fitted=model.fit(model.fit_inputs(rows,X,offsets,support,list(range(8)),'uniform'),.1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fold0-l2_0.1'
            runner.save_checkpoint(path,'fixed_binding',fitted,list(range(8)))
            restored=runner.load_checkpoint(path,'fixed_binding')
            np.testing.assert_array_equal(fitted['weights'],restored['weights'])
            np.testing.assert_array_equal(fitted['scale'],restored['scale'])
            original=model.corrections(fitted,X,offsets,list(range(8)))
            resumed=model.corrections(restored,X,offsets,list(range(8)))
            for a,b in zip(original,resumed):np.testing.assert_array_equal(a,b)
            with self.assertRaisesRegex(ValueError,'mismatch'):runner.load_checkpoint(path,'wrong_binding')
            receipt,weights=runner.checkpoint_paths(path)
            weights.write_bytes(weights.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'mismatch'):runner.load_checkpoint(path,'fixed_binding')

    def test_decimal_regularization_values_cannot_collide_in_cache(self):
        paths=[runner.checkpoint_paths(Path(f'fold-l2_{l2}')) for l2 in model.L2_GRID]
        self.assertEqual(len(set(paths)),4)
        self.assertEqual(str(paths[0][0]),'fold-l2_0.01.json')


if __name__=='__main__':unittest.main()
