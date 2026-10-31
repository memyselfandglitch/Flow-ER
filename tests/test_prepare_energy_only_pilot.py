"""Frozen solvent/pool joins, complete screening and distinct-group selection."""
import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_energy_only_pilot as prepare


def row(i, group, stratum):
    return {'source_row_id':i,'group':group,'stratum':stratum}


class FrozenProposalTests(unittest.TestCase):
    def test_join_rejects_changed_pool_and_source_without_reassigning_solvent(self):
        r = {'source_row_id':7,'source_line_sha256':'original',
             'candidates':[{'rank':1,'count':40,'smiles':'A'},{'rank':2,'count':24,'smiles':'B'}]}
        assignment = {'source_line_sha256':'original','candidate_pool_sha256':prepare.pilot.fingerprint(r['candidates']),
                      'status':'assumed','proposed_alpb_name':'thf'}
        before = copy.deepcopy(assignment)
        self.assertIs(prepare.join_assignment(r,{7:assignment}), assignment)
        r['candidates'][0]['count']=39
        with self.assertRaises(ValueError): prepare.join_assignment(r,{7:assignment})
        r['candidates'][0]['count']=40; r['source_line_sha256']='changed'
        with self.assertRaises(ValueError): prepare.join_assignment(r,{7:assignment})
        self.assertEqual(assignment,before)

    def test_starting_set_requires_recoverability_assumed_status_and_declared_mapping(self):
        r={'target_rank':2}; a={'status':'assumed','proposed_alpb_name':'thf'}
        self.assertTrue(prepare.in_starting_set(r,a))
        for rank in (None,1,11):
            self.assertFalse(prepare.in_starting_set({'target_rank':rank},a))
        self.assertFalse(prepare.in_starting_set(r,dict(a,status='unresolved')))
        self.assertFalse(prepare.in_starting_set(r,dict(a,proposed_alpb_name=None)))

    def test_augmenting_matching_avoids_artificial_shortfall(self):
        a,b=prepare.STRATA
        rows=[row(0,'shared',a),row(1,'a_only',a),row(2,'shared',b)]
        selected=prepare.select_distinct_groups(rows,quota=1)
        self.assertEqual(len(selected),2)
        self.assertEqual({r['stratum'] for r in selected},{a,b})
        self.assertEqual(len({r['group'] for r in selected}),2)

    def test_actual_shortage_does_not_relax_quota_or_repeat_groups(self):
        a,b=prepare.STRATA
        rows=[row(i,f'a{i}',a) for i in range(40)]+[row(100,'b',b),row(101,'b',b)]
        selected=prepare.select_distinct_groups(rows)
        self.assertEqual(sum(r['stratum']==a for r in selected),25)
        self.assertEqual(sum(r['stratum']==b for r in selected),1)
        self.assertEqual(len(selected),len({r['group'] for r in selected}))

    def test_selection_is_reproducible_independent_of_input_order_energy_or_rarity(self):
        a,b=prepare.STRATA
        rows=[row(i,f'g{i//2}',a if i%2 else b) for i in range(160)]
        original=prepare.select_distinct_groups(rows)
        changed=list(reversed(copy.deepcopy(rows)))
        for r in changed:r.update(energy=-999.,rarity=1,frequency=63)
        repeated=prepare.select_distinct_groups(changed)
        self.assertEqual([r['source_row_id'] for r in original],[r['source_row_id'] for r in repeated])
        self.assertEqual(len(original),50)
        self.assertEqual([sum(r['stratum']==s for r in original) for s in prepare.STRATA],[25,25])

    def test_all_candidates_are_checked_after_an_exception(self):
        r={'reactant':'unused','candidates':[{'rank':i,'smiles':str(i)} for i in range(1,5)]}
        def check(one):
            rank=one['candidates'][0]['rank']
            if rank==2:raise ValueError('unparseable')
            return [{'rank':rank,'reason':'size_limit'}] if rank==4 else []
        with patch.object(prepare.pilot,'static_pool_check',side_effect=check) as mocked:
            checks, failures=prepare.full_pool_screen(r)
        self.assertEqual(mocked.call_count,4)
        self.assertEqual([c['rank'] for c in checks],[1,2,3,4])
        self.assertEqual({f['reason'] for f in failures},{'unparseable_or_uncheckable_candidate','size_limit'})

    def test_planned_physical_limits_and_holds_unchanged(self):
        p=prepare.pilot.protocol('frozen')
        self.assertTrue(all(p['execution_holds'].values()))
        self.assertEqual(p['budget']['workers'],2)
        self.assertEqual(p['budget']['elapsed_seconds'],7200)
        self.assertEqual(p['physical']['max_heavy_atoms_per_fragment'],40)


if __name__=='__main__':unittest.main()
