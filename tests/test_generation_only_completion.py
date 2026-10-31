"""Completion must be proven before the rental becomes eligible for deletion."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finish_validation_generation_only import sha, verify_generation


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.experiment = self.root / 'experiment'
        self.pilot = self.experiment / 'inference'
        self.put('checkpoint', 'frozen weights', self.root)
        self.put('cache/val.jsonl', json.dumps({'sample_count': 64}) + '\n')
        self.put('inference/inputs/val000.txt', 'input>>target\n')
        self.put('inference/val000-rows.json', '[{}]\n')
        self.put('inference/predictions/val000/test-64-2940000_97.txt', 'saved prediction\n')
        self.put('inference/records/val000.jsonl', json.dumps({'sample_count': 64}) + '\n')
        source_hash = sha(self.pilot / 'inputs/val000.txt')
        rows_hash = sha(self.pilot / 'val000-rows.json')
        self.put('inference/selection.json', json.dumps({
            'parts': {'val000': {'source_split': 'val', 'rows': 1,
                                'inputs_sha256': source_hash, 'row_manifest_sha256': rows_hash}},
            'cached_validation_records_sha256': sha(self.experiment / 'cache/val.jsonl')}))
        self.put('protocol.json', json.dumps({
            'generator_checkpoint': {'path': 'checkpoint', 'sha256': sha(self.root / 'checkpoint')},
            'manifest_sha256': {}, 'training_rows': 2, 'reused_validation_rows': 1}))
        self.put('inference/inference-status.json', json.dumps({
            'status': 'completed', 'requested_parts': ['val000'], 'full_benchmark_prefix_unchanged': True}))
        self.put('inference/records/val000.manifest.json', json.dumps({
            'status': 'complete', 'rows': 1, 'sample_size': 64,
            'prediction_sha256': sha(self.pilot / 'predictions/val000/test-64-2940000_97.txt'),
            'records_sha256': sha(self.pilot / 'records/val000.jsonl'),
            'input_sha256': source_hash, 'row_manifest_sha256': rows_hash, 'logs_sha256': {}}))

    def put(self, name, value, root=None):
        path = (root or self.experiment) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def test_complete_generation_reports_training_held(self):
        result = verify_generation(self.root, self.experiment)
        self.assertEqual(result['validation_rows'], 2)
        self.assertFalse(result['reranker_training_started'])
        self.assertFalse(result['test_generation_started'])

    def test_incomplete_or_changed_output_is_rejected(self):
        self.put('inference/predictions/val000/test-64-2940000_97.txt', 'partial')
        with self.assertRaisesRegex(ValueError, 'Incomplete or changed'):
            verify_generation(self.root, self.experiment)

    def test_unexpected_training_is_reported(self):
        (self.experiment / 'model').mkdir()
        with self.assertRaisesRegex(ValueError, 'Unexpected new reranker'):
            verify_generation(self.root, self.experiment)

    def test_unexpected_test_generation_is_reported(self):
        self.put('inference/predictions/test000/test.txt', 'prediction\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected test generation'):
            verify_generation(self.root, self.experiment)


if __name__ == '__main__':
    unittest.main()
