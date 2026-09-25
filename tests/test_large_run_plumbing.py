"""No model inference: bounded residency, durable records, streaming exports."""
import json
from pathlib import Path
import tempfile
import unittest

from engine.bounded_batches import decoded_batches
from engine.record_index import RecordIndex


class LargeRunPlumbingTests(unittest.TestCase):
    def test_resume_cannot_mix_batch_or_source_identity(self):
        from engine.run_integrity import validate_resume_config
        with tempfile.TemporaryDirectory() as directory:
            config = {'batch_size': 32, 'source_manifest_sha256': 'original'}
            (Path(directory) / 'run_config.json').write_text(json.dumps(config))
            self.assertEqual(validate_resume_config(directory, config), config)
            for key, changed in [('batch_size', 64), ('source_manifest_sha256', 'changed')]:
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    validate_resume_config(directory, dict(config, **{key: changed}))

    def test_six_thousand_cases_keep_only_one_image_batch_resident(self):
        active = peak = 0
        class Image:
            def __init__(self, data):
                nonlocal active, peak
                active += 1; peak = max(peak, active)
            def close(self):
                nonlocal active
                active -= 1
        samples = ({'index': i, 'raw': {'image_bytes': b'x'}} for i in range(6000))
        observed = []
        for batch in decoded_batches(samples, 32, Image):
            observed.extend(s['index'] for s in batch)
            self.assertLessEqual(active, 32)
        self.assertEqual(observed, list(range(6000)))
        self.assertEqual(peak, 32)
        self.assertEqual(active, 0)

    def test_failed_decode_closes_only_owned_images(self):
        class Image:
            closed = False
            def close(self): self.closed = True
        owned, external = Image(), Image()
        def decode(data):
            if data == b'bad': raise ValueError('broken image')
            return owned
        rows = [{'image': external}, {'raw': {'image_bytes': b'ok'}}, {'raw': {'image_bytes': b'bad'}}]
        with self.assertRaises(ValueError):
            list(decoded_batches(rows, 3, decode))
        self.assertTrue(owned.closed)
        self.assertFalse(external.closed)

    def test_index_retains_metadata_only_and_validates_repeated_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            index = RecordIndex(directory)
            for i in reversed(range(64)):
                index[str(i)] = {'id': str(i), 'sample': i, 'trace': {'large': 'x' * 8192}}
            self.assertLess(len(repr(index.entries)), 10000)
            rows = index.ordered(list(index))
            self.assertEqual([r['sample'] for r in rows], list(range(64)))
            self.assertEqual([r['sample'] for r in rows], list(range(64)))
            restored = RecordIndex(directory, resume=True)
            self.assertEqual(len(restored), 64)
            path = restored._path('3')
            payload = json.loads(path.read_text()); payload['record']['sample'] = 99
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                restored['3']

    def test_legacy_journal_migration_detects_conflicts_and_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / 'records.jsonl'
            row = {'id': 'one', 'sample': 1}
            journal.write_text(json.dumps(row) + '\n{"id":')
            index = RecordIndex(directory, resume=True)
            self.assertEqual(index['one'], row)
            journal.write_text(json.dumps(dict(row, sample=2)) + '\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                RecordIndex(directory, resume=True)

    def test_exports_accept_disk_backed_reiterable_records(self):
        from run_figdebate import finalize_saved_run
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            index = RecordIndex(directory)
            value = {'id': 'synthetic', 'sample': 0, 'initial_prediction': 'ENTAILS',
                     'prediction': 'ENTAILS', 'ground_truth': 'ENTAILS', 'judge_requested': False}
            index[value['id']] = value
            with patch('run_figdebate.evaluate_predictions', return_value={'accuracy': 1.0}):
                result = finalize_saved_run(directory, [value['id']], index.ordered([value['id']]), {}, {})
            self.assertEqual(result[0]['accuracy'], 1.)
            self.assertTrue((Path(directory) / 'completion_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
