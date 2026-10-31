import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_full_validation import prepare_metadata
from prepare_full_inference import write_shards


def row(split, index, reactant, network):
    return dict(source_split=split, source_row_id=index, reactant_key=reactant,
                sequence_id=f'{split}:{network}')


class FullValidationPreparationTests(unittest.TestCase):
    def test_every_row_retained_and_connected_validation_rows_share_fold(self):
        val = [row('val', 0, 'a', 1), row('val', 1, 'b', 1),
               row('val', 2, 'b', 2), row('val', 3, 'c', 3)]
        test = [row('test', 0, 'z', 1), row('test', 1, 'b', 2),
                row('test', 2, 'q', 2), row('test', 3, 'v', 3)]
        original = copy.deepcopy((val, test))
        fitted, assessed, summary = prepare_metadata(val, test, 1)
        self.assertEqual((val, test), original)
        self.assertEqual(len(fitted), 4)
        self.assertEqual(len(assessed), 4)
        self.assertEqual(len({r['cv_fold'] for r in fitted[:3]}), 1)
        self.assertEqual(summary['test_exact_input_overlap_with_validation'], 1)
        self.assertEqual(summary['test_group_overlap_with_validation'], 2)
        self.assertTrue(assessed[0]['previously_inspected'])
        self.assertFalse(assessed[0]['group_seen_in_validation'])
        self.assertTrue(assessed[3]['fresh_group_disjoint'])
        self.assertEqual(summary['fresh_group_disjoint_test_rows'], 1)

    def test_test_network_connections_cannot_split_duplicate_training_inputs(self):
        val = [row('val', 0, 'a', 1), row('val', 1, 'b', 2)]
        test = [row('test', 0, 'a', 8), row('test', 1, 'b', 8)]
        fitted, _, _ = prepare_metadata(val, test, 0)
        self.assertEqual(fitted[0]['split_group'], fitted[1]['split_group'])
        self.assertEqual(fitted[0]['cv_fold'], fitted[1]['cv_fold'])

    def test_incomplete_duplicate_or_wrong_source_is_rejected(self):
        good = [row('val', 0, 'a', 1), row('val', 1, 'b', 2)]
        test = [row('test', 0, 'c', 1)]
        for bad in (good[1:], [good[0], good[0]], [dict(good[0], source_split='test')]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare_metadata(bad, test, 0)

    def test_previously_inspected_network_excluded_only_from_fresh_stratum(self):
        val = [row('val', 0, 'a', 1)]
        test = [row('test', 0, 'b', 1), row('test', 1, 'c', 1)]
        _, assessed, summary = prepare_metadata(val, test, 1)
        self.assertFalse(assessed[1]['previously_inspected'])
        self.assertTrue(assessed[1]['group_touches_inspected_test'])
        self.assertFalse(assessed[1]['fresh_group_disjoint'])
        self.assertEqual(summary['test_rows'], 2)
        self.assertEqual(summary['excluded_test_rows'], 0)

    def test_shards_preserve_all_rows_in_order_with_global_and_local_indices(self):
        lines = [f'input-{i}>>target-{i}|{i}\n'.encode() for i in range(5)]
        metadata = [dict(source_split='val', source_row_id=i,
                         source_line_sha256=hashlib.sha256(line).hexdigest())
                    for i, line in enumerate(lines)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / 'source.txt'
            source.write_bytes(b''.join(lines))
            (output / 'inputs').mkdir()
            parts = write_shards(source, metadata, output, 'val', 2)
            self.assertEqual([p['rows'] for p in parts.values()], [2, 2, 1])
            self.assertEqual(b''.join((output / 'inputs' / f'{name}.txt').read_bytes()
                                      for name in parts), source.read_bytes())
            last = json.loads((output / 'val002-rows.json').read_text())
            self.assertEqual((last[0]['source_row_id'], last[0]['row_id']), (4, 0))

    def test_shards_reject_source_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / 'source.txt'
            source.write_bytes(b'wrong\n')
            (output / 'inputs').mkdir()
            metadata = [dict(source_split='val', source_row_id=0,
                             source_line_sha256=hashlib.sha256(b'expected\n').hexdigest())]
            with self.assertRaisesRegex(ValueError, 'source alignment'):
                write_shards(source, metadata, output, 'val', 2)


if __name__ == '__main__':
    unittest.main()
