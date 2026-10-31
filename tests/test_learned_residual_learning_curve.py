"""Whole-group subsampling, subset-only fitting and paired-repeat inference."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy import sparse
import learned_residual_learning_curve as curve


def fixture():
    rows=[]
    for g in range(18):
        for member in range(1+g%3):
            i=len(rows)
            rows.append({'source_row_id':i,'group':f'g{g}','fold':g%3,
                'normalized_input':f'n{g}','exact_input':f'x{g}','target':'B' if member%2 else 'A',
                'candidates':[{'smiles':'A','count':40},{'smiles':'B','count':24}]})
    configs={i:{'l2':.1,'weighting':'uniform','lambda':1.,'margin':.1} for i in range(3)}
    X=sparse.csr_matrix(np.array([[i%2,(i//2)%3,(i%7)-3] for i in range(2*len(rows))],dtype=float))
    return rows,configs,X,np.arange(0,2*len(rows)+1,2),[True]*len(rows)


class GroupSamplingTests(unittest.TestCase):
    def test_subsets_nested_whole_group_and_identical_held_rows(self):
        rows,policies,_,_,_=fixture();manifest=curve.build_manifest(rows,policies,(11,12))
        self.assertEqual(len(manifest['jobs']),15)
        curve.verify_manifest(rows,manifest,policies)
        jobs={j['id']:j for j in manifest['jobs']}
        for outer in policies:
            full=jobs[f'outer{outer}-pct100']
            self.assertEqual(full['available_groups'],12)
            for seed in (11,12):
                small=jobs[f'outer{outer}-pct25-seed{seed}'];medium=jobs[f'outer{outer}-pct50-seed{seed}']
                self.assertEqual(len(small['fit_groups']),3);self.assertEqual(len(medium['fit_groups']),6)
                self.assertTrue(set(small['fit_ids'])<set(medium['fit_ids'])<set(full['fit_ids']))
                self.assertEqual(small['held_ids'],medium['held_ids']);self.assertEqual(small['held_ids'],full['held_ids'])
                for job in (small,medium,full):
                    expected={i for i,row in enumerate(rows) if row['group'] in job['fit_groups']}
                    self.assertEqual(set(job['fit_ids']),expected)
                    self.assertFalse(set(job['fit_ids'])&set(job['held_ids']))

    def test_sampling_is_deterministic_label_blind_and_seed_sensitive(self):
        rows,policies,_,_,_=fixture()
        original=curve.build_manifest(rows,policies,(11,12))
        changed=deepcopy(rows)
        for row in changed:row['target']='CHANGED';row['target_rank']=99;row['baseline_correct']=False
        self.assertEqual(original,curve.build_manifest(changed,policies,(11,12)))
        a={j['outer_fold']:j['fit_groups'] for j in original['jobs'] if j['fraction_percent']==25 and j['subset_seed']==11}
        b={j['outer_fold']:j['fit_groups'] for j in original['jobs'] if j['fraction_percent']==25 and j['subset_seed']==12}
        self.assertNotEqual(a,b)

    def test_group_crossing_or_changed_manifest_is_rejected(self):
        rows,policies,_,_,_=fixture()
        manifest=curve.build_manifest(rows,policies,(11,))
        changed=deepcopy(manifest);changed['jobs'][1]['fit_ids'].pop()
        with self.assertRaisesRegex(ValueError,'manifest changed'):curve.verify_manifest(rows,changed,policies)
        rows[1]['normalized_input']=rows[0]['normalized_input']
        with self.assertRaisesRegex(ValueError,'Fold leakage'):curve.build_manifest(rows,policies,(11,))

    def test_fixed_policy_has_no_outer_held_labels_in_selection(self):
        rows,policies,_,_,_=fixture();outer=1
        selection={'selection_source_row_ids':[r['source_row_id'] for r in rows if r['fold']!=outer],
            'selected':{'config':policies[outer]}}
        self.assertEqual(curve.policy_from_selection(rows,outer,selection),policies[outer])
        selection['selection_source_row_ids'].append(next(r['source_row_id'] for r in rows if r['fold']==outer))
        with self.assertRaisesRegex(ValueError,'exclusively'):curve.policy_from_selection(rows,outer,selection)


class FitBoundaryTests(unittest.TestCase):
    def test_omitted_fitting_and_held_labels_cannot_change_subset_model(self):
        rows,policies,X,offsets,support=fixture()
        job=next(j for j in curve.build_manifest(rows,policies,(11,))['jobs'] if j['fraction_percent']==25)
        changed=deepcopy(rows);selected=set(job['fit_ids'])
        for i,row in enumerate(changed):
            if i not in selected:row['target']='outside_subset_reference_CHANGED'
        a=curve.fresh_fit(rows,X,offsets,support,job)
        b=curve.fresh_fit(changed,X,offsets,support,job)
        for key in ('weights','scale'):np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a['info'],b['info']);self.assertTrue(a['info']['initialized_from_zero'])
        self.assertEqual(a['info']['fitting_rows'],len(selected))

    def test_nonconvergence_cannot_trigger_replacement_or_scoring(self):
        rows,policies,X,offsets,support=fixture()
        job=curve.build_manifest(rows,policies,(11,))['jobs'][1]
        before=deepcopy(job)
        with patch.object(curve.model,'fit',return_value={'converged':False}):
            with self.assertRaisesRegex(RuntimeError,'no replacement subset'):
                curve.fresh_fit(rows,X,offsets,support,job)
        self.assertEqual(job,before)


class AssessmentTests(unittest.TestCase):
    def test_missing_duplicate_or_mismatched_groups_are_rejected(self):
        rows,_,_,_,_=fixture()
        outcomes=[{'source_row_id':r['source_row_id'],'group':r['group'],'fold':r['fold']} for r in rows]
        self.assertEqual(curve.aligned_outcomes(rows,list(reversed(outcomes))),outcomes)
        for bad in (outcomes[:-1],outcomes+[outcomes[0]]):
            with self.assertRaisesRegex(ValueError,'Missing or duplicated'):curve.aligned_outcomes(rows,bad)
        bad=deepcopy(outcomes);bad[0]['group']='wrong'
        with self.assertRaisesRegex(ValueError,'group mismatch'):curve.aligned_outcomes(rows,bad)

    def test_repeated_subsets_do_not_multiply_independent_groups(self):
        groups=['small']+['large']*9
        differences=np.array([[1]+[0]*9]*5)
        captured={}
        def bootstrap(sizes,totals,rng,replicates):
            captured.update(sizes=sizes,totals=totals,replicates=replicates)
            return np.array([.1,.1,.1])
        with patch.object(curve,'bootstrap_ratios',side_effect=bootstrap):
            result=curve.cluster_interval(groups,differences,replicates=3)
        self.assertEqual(result['groups'],2);self.assertEqual(result['unique_assessment_rows'],10)
        self.assertEqual(result['subset_repeats'],5);self.assertAlmostEqual(result['change'],.1)
        self.assertEqual(len(captured['sizes']),2)
        self.assertEqual(sorted(captured['sizes'].tolist()),[5,45])
        self.assertEqual(captured['totals'].sum(),5)

    def test_bootstrap_is_invariant_to_exact_duplication_of_repeat(self):
        groups=['a','b','b','c','c','c']
        one=np.array([[1,-1,0,0,0,1]])
        a=curve.cluster_interval(groups,one,replicates=200)
        b=curve.cluster_interval(groups,np.repeat(one,5,axis=0),replicates=200)
        self.assertEqual(a['change'],b['change']);np.testing.assert_array_equal(a['ci95'],b['ci95'])

    def test_fractional_differences_cannot_be_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'binary-outcome'):
            curve.cluster_interval(['a','b'],[[0.,.5]],replicates=3)

    def test_changed_prior_artifact_fails_preservation_check(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'model').write_text('original')
            expected=curve.snapshot(p);curve.verify_snapshot(p,expected)
            (p/'model').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Original experiment changed'):curve.verify_snapshot(p,expected)


if __name__=='__main__':unittest.main()
