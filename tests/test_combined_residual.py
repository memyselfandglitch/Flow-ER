"""Cross-source grouping, protected outcomes and immutable candidate checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_combined_residual as runner


def fixture():
    return [{'row_uid':f'val:{i}','source_split':'val','source_row_id':i,
        'base_group':f'b{i}','exact_input':f'e{i}','normalized_input':f'n{i}',
        'raw_sequence_id':'RC','target':'A'} for i in range(100)]


class GroupTests(unittest.TestCase):
    def test_identical_source_row_numbers_in_different_splits_are_distinct(self):
        rows=fixture();rows[1].update(row_uid='test:0',source_split='test',source_row_id=0)
        runner.group_rows(rows)
        self.assertNotEqual(rows[0]['group'],rows[1]['group'])

    def test_links_across_sources_are_transitive(self):
        rows=fixture();rows[1].update(row_uid='test:1',source_split='test',normalized_input='n0')
        rows[2]['exact_input']=rows[1]['exact_input'];rows[3]['base_group']=rows[2]['base_group']
        runner.group_rows(rows)
        self.assertEqual(len({r['group'] for r in rows[:4]}),1)
        self.assertEqual(len({r['fold'] for r in rows[:4]}),1)

    def test_numeric_networks_are_source_local_and_generic_tags_do_not_group(self):
        rows=fixture();rows[0]['raw_sequence_id']='7';rows[1]['raw_sequence_id']='7'
        rows[2].update(source_split='test',row_uid='test:2',raw_sequence_id='7')
        runner.group_rows(rows)
        self.assertEqual(rows[0]['group'],rows[1]['group'])
        self.assertNotEqual(rows[0]['group'],rows[2]['group'])
        self.assertNotEqual(rows[3]['group'],rows[4]['group'])

    def test_group_assignment_is_order_and_label_independent(self):
        first=fixture();second=list(reversed(deepcopy(first)))
        for r in second:r['target']='changed'
        runner.group_rows(first);runner.group_rows(second)
        self.assertEqual({r['row_uid']:(r['group'],r['fold']) for r in first},
                         {r['row_uid']:(r['group'],r['fold']) for r in second})

    def test_reserved_overlap_rejected_for_each_identity(self):
        rows=fixture()
        base={'source_split':'test','source_row_id':999,'reactant_key':'other','split_group':'other'}
        self.assertEqual(runner.reserved_guard(rows,[base])['overlap'],{'rows':0,'normalized_inputs':0,'base_components':0})
        for update in ({'source_split':'val','source_row_id':0},{'reactant_key':'n0'},{'split_group':'b0'}):
            with self.assertRaisesRegex(ValueError,'Reserved test overlap'):
                runner.reserved_guard(rows,[dict(base,**update)])

    def test_reserved_outcome_files_cannot_be_opened(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('must not open')):
            for relative in ('data/flower_new_dataset/test.txt',
                'results/reranking/full-validation-test20k-20261029/inference/inputs/test000.txt'):
                with self.assertRaises(PermissionError):runner.read_allowed(runner.ROOT/relative)


class PoolTests(unittest.TestCase):
    def test_top10_prefix_preserves_counts_order_and_source(self):
        row={'sample_count':64,'target':'c10','target_rank':11,
            'candidates':[{'rank':i+1,'count':2,'smiles':f'c{i}','old_feature':'ignored'} for i in range(11)]}
        before=deepcopy(row);cs=runner.checked_pool(row)
        self.assertEqual(row,before);self.assertEqual(len(cs),10)
        self.assertEqual(cs[-1],{'rank':10,'count':2,'smiles':'c9'})

    def test_rank_or_count_corruption_rejected(self):
        row={'sample_count':64,'target':'B','target_rank':2,
             'candidates':[{'rank':1,'count':40,'smiles':'A'},{'rank':2,'count':24,'smiles':'B'}]}
        runner.checked_pool(row)
        for field,value in [('count',50),('rank',1)]:
            changed=deepcopy(row);changed['candidates'][1][field]=value
            with self.assertRaises(ValueError):runner.checked_pool(changed)
        changed=deepcopy(row);changed['target_rank']=1
        with self.assertRaises(ValueError):runner.checked_pool(changed)


if __name__=='__main__':unittest.main()
