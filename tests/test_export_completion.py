"""Public finalization path: durable inference survives derived-export failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from engine.result_store import commit_record, read_records
from run_figdebate import finalize_saved_run


class ExportCompletionTests(unittest.TestCase):
    def test_export_failure_and_retry_preserve_exact_committed_result(self):
        record = {'id': 'synthetic', 'sample': 0, 'initial_prediction': 'ENTAILS',
                  'prediction': 'ENTAILS', 'ground_truth': 'ENTAILS', 'judge_requested': False}
        for failing_export in ('write_predictions', 'write_debate_logs', 'write_feedback_decision_logs', 'evaluate_predictions'):
            with self.subTest(export=failing_export), tempfile.TemporaryDirectory() as directory:
                commit_record(directory, record)
                progress = {'status': 'running'}
                with patch('run_figdebate.' + failing_export, side_effect=PermissionError('locked')):
                    with self.assertRaises(PermissionError):
                        finalize_saved_run(directory, ['synthetic'], [record], {}, progress)
                self.assertEqual(read_records(Path(directory, 'records.jsonl')), {'synthetic': record})
                self.assertEqual(json.loads(Path(directory, 'progress.json').read_text())['status'],
                                 'inference_complete_export_pending')
                self.assertFalse(Path(directory, 'completion_manifest.json').exists())
                with patch('run_figdebate.evaluate_predictions', return_value={'accuracy': 1.0}):
                    finalize_saved_run(directory, ['synthetic'], [record], {}, progress)
                self.assertEqual(progress['status'], 'complete')
                self.assertNotIn('failure_reason', progress)
                self.assertEqual(json.loads(Path(directory, 'completion_manifest.json').read_text())['samples'], 1)

    def test_timing_export_failure_also_marks_export_pending(self):
        from engine.result_store import atomic_json
        record = {'id': 'synthetic'}
        with tempfile.TemporaryDirectory() as directory:
            commit_record(directory, record)
            def write(path, value):
                if Path(path).name == 'run_timing.json':
                    raise PermissionError('timing export locked')
                return atomic_json(path, value)
            with patch('run_figdebate.write_json_atomic', side_effect=write):
                with self.assertRaises(PermissionError):
                    finalize_saved_run(directory, ['synthetic'], [record], {}, {})
            self.assertEqual(json.loads(Path(directory, 'progress.json').read_text())['failure_phase'], 'export')
            self.assertEqual(read_records(Path(directory, 'records.jsonl')), {'synthetic': record})
