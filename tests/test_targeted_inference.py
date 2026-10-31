import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_targeted_inference as runner


def fixture_row(index=0, source='val', smiles='[CH4:1]'):
    line = f'{smiles}>>{smiles}|{index + 7}\n'
    meta = {'row_id': index, 'source_row_id': 900 + index, 'source_split': source,
            'sequence_id': f'{source}:{index + 7}', 'raw_sequence_id': str(index + 7),
            'split_group': 'group-' + str(index),
            'source_line_sha256': runner.digest_bytes(line.encode()),
            'reactant_key': runner.digest_bytes(runner.canonical(smiles).encode())}
    target = runner.reference(smiles)
    prediction = f'[64, 0, 0, 0, 0]|0|{[(target, 64, True)]!r}\n'.encode()
    return line, meta, prediction


class ExtractionTests(unittest.TestCase):
    def test_extracted_audit_preserves_frozen_cohort_membership(self):
        import targeted_reranker
        truth, meta, prediction = fixture_row()
        meta['selection_cohort'] = 'random_audit'
        row = runner.extract_row(truth, prediction, 0, meta)
        self.assertEqual(row['selection_cohort'], 'random_audit')
        targeted_reranker.validate_records([row], 'val')
        targeted_reranker.manifest_membership([row], [meta])

    def test_raw_opaque_candidate_and_tie_order_are_preserved(self):
        truth, meta, _ = fixture_row(source='train')
        target = runner.reference('[CH4:1]')
        raw = f'[32, 32, 0, 0, 0]|1|{[("opaque!not-smiles", 32, True), (target, 32, True)]!r}\n'.encode()
        row = runner.extract_row(truth, raw, 0, meta)
        self.assertEqual(row['candidates'][0]['smiles'], 'opaque!not-smiles')
        self.assertEqual(row['candidates'][0]['rank'], 1)
        self.assertFalse(row['candidates'][0]['parseable'])
        self.assertFalse(row['rerank_eligible'])
        self.assertEqual(row['unsupported_candidate_ranks'], [1])
        self.assertEqual(row['target_rank'], 2)
        self.assertEqual(row['source_split'], 'train')
        self.assertEqual(row['source_row_id'], 900)
        self.assertEqual(row['row_id'], 0)
        self.assertEqual(row['saved_asymmetric_matrix_count'], 1)

    def test_no_saved_candidates_is_a_retained_row(self):
        truth, meta, _ = fixture_row()
        row = runner.extract_row(truth, b'[0, 0, 0, 64, 0]|0|[]\n', 0, meta)
        self.assertEqual(row['candidates'], [])
        self.assertIsNone(row['target_rank'])
        self.assertEqual(row['sample_count'], 64)
        self.assertEqual(row['source_split'], 'val')

    def test_invalid_sample_counts_or_target_counts_fail(self):
        truth, meta, _ = fixture_row()
        for prediction in (b'[0, 0, 0, 63, 0]|0|[]\n',
                           b"[1, 63, 0, 0, 0]|0|[('opaque!', 64, True)]\n"):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                runner.extract_row(truth, prediction, 0, meta)

    def test_partial_line_and_excess_rows_are_preserved_and_rejected(self):
        truth, meta, raw = fixture_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.txt'
            path.write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError, 'incomplete final'):
                runner.verified_prefix(path, [truth], [meta])
            self.assertEqual(path.read_bytes(), raw[:-1])
            path.write_bytes(raw + raw)
            with self.assertRaisesRegex(ValueError, 'More predictions'):
                runner.verified_prefix(path, [truth], [meta])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.pilot = self.root / 'pilot'
        for name in ('inputs', 'records', 'attempts', 'logs'):
            (self.pilot / name).mkdir(parents=True, exist_ok=True)
        examples = [fixture_row(0), fixture_row(1, smiles='[OH2:1]')]
        self.truth = [x[0] for x in examples]
        self.metadata = [x[1] for x in examples]
        self.predictions = [x[2] for x in examples]
        inputs = self.pilot / 'inputs' / 'fit.txt'
        inputs.write_text(''.join(self.truth))
        manifest = self.pilot / 'fit-rows.json'
        manifest.write_text(json.dumps(self.metadata))
        self.spec = {'rows': 2, 'source_split': 'val',
                     'inputs_sha256': runner.sha256(inputs),
                     'row_manifest_sha256': runner.sha256(manifest)}
        self.config = {'environment': {}}
        self.state = {}

    def tearDown(self):
        self.directory.cleanup()

    def launch(self, complete):
        def fake(command, cwd, env, log_path, deadline, on_start, on_progress):
            path = Path(env['RESULT_PATH']) / runner.PREDICTION_NAME
            prefix = len(runner.raw_prediction_lines(path))
            with log_path.open('x') as log:
                log.write(f'Continuing previous runs at reaction {prefix}...\n' if path.exists()
                          else 'Starting new run...\n')
                log.write('0.0% data is unparseable\n')
            with path.open('ab') as stream:
                stream.write(b''.join(self.predictions[prefix:2 if complete else 1]))
            on_progress()
            if not complete:
                raise TimeoutError('simulated interruption')
        return fake

    def execute(self, launch):
        return runner.execute_part(self.root, self.pilot, 'fit', self.spec, 'frozen-id',
                                   self.config, time.monotonic() + 10, self.state,
                                   lambda: None, launch=launch)

    def test_resume_appends_only_remaining_rows_and_reuses_verified_result(self):
        with self.assertRaises(TimeoutError):
            self.execute(self.launch(False))
        prediction = self.pilot / 'predictions' / 'fit' / runner.PREDICTION_NAME
        self.assertEqual(prediction.read_bytes(), self.predictions[0])
        receipt = self.execute(self.launch(True))
        self.assertEqual(prediction.read_bytes(), b''.join(self.predictions))
        self.assertEqual(receipt['rows'], 2)
        records = [json.loads(line) for line in (self.pilot / 'records' / 'fit.jsonl').read_text().splitlines()]
        self.assertEqual([row['source_row_id'] for row in records], [900, 901])
        self.assertEqual([row['source_split'] for row in records], ['val', 'val'])
        self.assertEqual(self.execute(lambda *args: self.fail('completed part was rerun')), receipt)

    def test_changed_input_fails_before_launch(self):
        (self.pilot / 'inputs' / 'fit.txt').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
            self.execute(lambda *args: self.fail('changed input was launched'))

    def test_resumed_process_cannot_replace_verified_prefix(self):
        with self.assertRaises(TimeoutError):
            self.execute(self.launch(False))
        complete = self.launch(True)
        def mutate(*args):
            complete(*args)
            path = self.pilot / 'predictions' / 'fit' / runner.PREDICTION_NAME
            path.write_bytes(path.read_bytes().replace(b'|0|', b'|1|', 1))
        with self.assertRaisesRegex(ValueError, 'prefix changed during inference'):
            self.execute(mutate)
        self.assertFalse((self.pilot / 'records' / 'fit.jsonl').exists())

    def test_completed_log_mutation_is_rejected(self):
        self.execute(self.launch(True))
        log = self.pilot / 'logs' / 'inference-fit-attempt-0001.log'
        log.write_text(log.read_text() + 'unexpected change\n')
        with self.assertRaisesRegex(ValueError, 'logs changed'):
            self.execute(lambda *args: self.fail('completed part was relaunched'))

    def test_foreign_predictions_and_changed_saved_prefix_are_rejected(self):
        path = self.pilot / 'predictions' / 'fit' / runner.PREDICTION_NAME
        path.parent.mkdir(parents=True)
        path.write_bytes(self.predictions[0])
        with self.assertRaisesRegex(ValueError, 'foreign predictions'):
            self.execute(self.launch(True))
        path.unlink()
        with self.assertRaises(TimeoutError):
            self.execute(self.launch(False))
        # Valid metric syntax, different bytes: the receipt must still catch it.
        path.write_bytes(self.predictions[0].replace(b'|0|', b'|1|'))
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            self.execute(self.launch(True))

    def test_running_receipt_is_sealed_before_next_attempt(self):
        with self.assertRaises(TimeoutError):
            self.execute(self.launch(False))
        receipt_path = self.pilot / 'attempts' / 'fit-0001.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['status'] = 'running'
        receipt.pop('end_rows')
        receipt.pop('prediction_sha256')
        receipt_path.write_text(json.dumps(receipt))
        self.execute(self.launch(True))
        self.assertEqual(json.loads(receipt_path.read_text())['status'], 'interrupted_prefix_verified')
        self.execute(lambda *args: self.fail('completed part was rerun'))

    def test_default_parts_exclude_holdout(self):
        selection = {'parts': {name: {} for name in ('fit', 'tune', 'audit', 'holdout')}}
        self.assertEqual(runner.selected_parts(selection, None), ['fit', 'tune', 'audit'])
        self.assertEqual(runner.selected_parts(selection, 'holdout'), ['holdout'])
        with self.assertRaises(ValueError):
            runner.selected_parts(selection, '../holdout')

    def test_identity_changes_and_unrequested_resume_fail(self):
        identity = {'config': {'batch_size': 512}, 'source_sha256': 'original'}
        runner.freeze_or_verify_identity(self.pilot, identity, False)
        with self.assertRaisesRegex(ValueError, 'use --resume'):
            runner.freeze_or_verify_identity(self.pilot, identity, False)
        runner.freeze_or_verify_identity(self.pilot, identity, True)
        for changed in ({'config': {'batch_size': 2048}, 'source_sha256': 'original'},
                        {'config': {'batch_size': 512}, 'source_sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                runner.freeze_or_verify_identity(self.pilot, changed, True)

    def test_logs_require_zero_skips_and_correct_resume_offset(self):
        path = self.pilot / 'logs' / 'alignment.log'
        for text in ('Starting new run...\n',
                     'Starting new run...\n0.1% data is unparseable\n',
                     'Continuing previous runs at reaction 1...\n0.0% data is unparseable\n'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.verify_log(path, 0, 1)
        path.write_text('Continuing previous runs at reaction 1...\n0.0% data is unparseable\n')
        runner.verify_log(path, 1, 1)

    def test_timeout_terminates_process_group(self):
        # Install a child signal handler that leaves evidence before exiting.
        child = self.root / 'child.py'
        ready, stopped = self.root / 'ready', self.root / 'stopped'
        child.write_text('import signal,time,sys\nfrom pathlib import Path\n'
                         'def stop(sig,frame):\n Path(sys.argv[2]).write_text("stopped");sys.exit(0)\n'
                         'signal.signal(signal.SIGTERM,stop)\nPath(sys.argv[1]).write_text("ready")\n'
                         'while True: time.sleep(0.02)\n')
        command = [sys.executable, '-c',
                   'import subprocess,time,sys,signal\np=subprocess.Popen(sys.argv[1:])\n'
                   'def stop(sig,frame):\n p.wait(timeout=3);sys.exit(0)\n'
                   'signal.signal(signal.SIGTERM,stop)\n'
                   'while True: time.sleep(0.02)\n',
                   sys.executable, str(child), str(ready), str(stopped)]
        pid = []
        with self.assertRaises(TimeoutError):
            runner.run_process(command, self.root, os.environ.copy(), self.root / 'process.log',
                               time.monotonic() + 0.7, pid.append, lambda: None)
        self.assertTrue(ready.exists())
        self.assertTrue(stopped.exists(), 'child did not receive group termination')


if __name__ == '__main__':
    unittest.main()
