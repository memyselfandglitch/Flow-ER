import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import resume_full_validation_experiment as resume


class ResumeTests(unittest.TestCase):
    def test_resume_keeps_inference_arguments_and_uses_a_new_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inference').mkdir()
            (root / 'inference/inference-status.json').write_text(json.dumps({
                'status': 'completed', 'requested_parts': ['val000', 'val001']}))
            calls = []
            run = resume.resume_stages(root, 'resume-1', lambda *args: calls.append(args))
            command = ['python', 'scripts/run_full_inference.py', '--parts', 'val000,val001']
            run(command, root / 'val-inference.log', 100)
            self.assertEqual(calls[0][0], command + ['--resume'])
            self.assertEqual(calls[0][1].name, 'val-inference-resume-1.log')
            self.assertNotIn('--resume', command)

    def test_failed_inference_receipt_blocks_training_even_with_zero_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inference').mkdir()
            (root / 'inference/inference-status.json').write_text(json.dumps({
                'status': 'failed_or_incomplete', 'requested_parts': ['val003']}))
            run = resume.resume_stages(root, 'resume-1', lambda *args: None)
            with self.assertRaisesRegex(RuntimeError, 'did not finish'):
                run(['python', 'scripts/run_full_inference.py', '--parts', 'val003'], root / 'log', 100)

    def test_existing_frozen_reranker_is_verified_and_never_refitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'model').mkdir()
            calls = []
            run = resume.resume_stages(root, 'resume-1', lambda *args: calls.append(args))
            with patch.object(resume, 'verify_existing_model') as verify:
                run(['python', 'scripts/full_validation_reranker.py', 'fit'], root / 'fit.log', 100)
            verify.assert_called_once_with(root)
            self.assertEqual(calls, [])

    def test_interrupted_attempt_is_archived_and_sealed_without_changing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); experiment = root / 'experiment'; pilot = experiment / 'inference'
            history = experiment / 'recovery' / 'resume-1'
            history.mkdir(parents=True)
            for name in ('attempts', 'logs', 'predictions/val000'):
                (pilot / name).mkdir(parents=True)
            identity = {'config': {}, 'preserved_prediction': '/preserved',
                        'preserved_prediction_sha256': 'preserved-hash'}
            (pilot / 'inference-identity.json').write_text(json.dumps(identity))
            identity_hash = resume.base.sha256(pilot / 'inference-identity.json')
            (pilot / 'selection.json').write_text(json.dumps({'parts': {'val000': {}}}))
            prediction = pilot / 'predictions/val000' / resume.base.PREDICTION_NAME
            prediction.write_bytes(b'prediction-one\nprediction-two\n')
            before = prediction.read_bytes()
            attempt = {'status': 'running', 'identity_sha256': identity_hash, 'start_rows': 0,
                       'prefix_sha256': resume.base.digest_bytes(b''),
                       'log': 'logs/attempt.log', 'pid': 1}
            attempt_path = pilot / 'attempts/val000-0001.json'
            attempt_path.write_text(json.dumps(attempt))
            (pilot / 'logs/attempt.log').write_text('0.0% data is unparseable\nStarting new run...\n')
            (experiment / 'run-status.json').write_text('{"status":"running"}')
            (experiment / 'ready-for-backup.json').write_text('{"status":"incomplete"}')
            with patch.object(resume.inference, 'identity', return_value=identity), \
                    patch.object(resume.base, 'load_part', return_value=(['a', 'b'], [])), \
                    patch.object(resume.base, 'verified_prefix', return_value=(
                        before.splitlines(keepends=True), [{'sample_count': 64}, {'sample_count': 64}])):
                receipt = resume.preserve_and_verify(root, experiment, history)
            self.assertEqual(prediction.read_bytes(), before)
            self.assertEqual(receipt['rows'], {'val000': 2})
            self.assertEqual(json.loads((history / attempt_path.name).read_text()), attempt)
            sealed = json.loads(attempt_path.read_text())
            self.assertEqual(sealed['status'], 'interrupted_prefix_verified')
            self.assertEqual(sealed['end_rows'], 2)
            self.assertFalse((experiment / 'ready-for-backup.json').exists())
            self.assertTrue((history / 'ready-for-backup.json').exists())


if __name__ == '__main__':
    unittest.main()
