import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from engine.result_store import atomic_json, commit_record, read_records, completion_manifest, replace_with_retry


class ResultStoreTests(unittest.TestCase):
    def test_committed_result_survives_incomplete_journal(self):
        with tempfile.TemporaryDirectory() as d:
            record={'id':'case','prediction':'ENTAILS'}
            commit_record(d,record)
            Path(d,'records.jsonl').write_bytes(b'{"id":')
            self.assertEqual(read_records(Path(d,'records.jsonl')),{'case':record})
            commit_record(d,record)
            with self.assertRaises(ValueError): commit_record(d,dict(record,prediction='CONTRADICTS'))

    def test_complete_corruption_and_conflicting_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'records.jsonl'); p.write_text('{broken}\n')
            with self.assertRaises(ValueError): read_records(p)
            p.write_text('{"id":"a","v":1}\n{"id":"a","v":2}\n')
            with self.assertRaises(ValueError): read_records(p)

    def test_tampered_commit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            commit_record(d,{'id':'a'})
            p=next(Path(d,'final_records').glob('*.json')); value=json.loads(p.read_text()); value['record']['id']='b'; p.write_text(json.dumps(value))
            with self.assertRaises(ValueError): read_records(Path(d,'records.jsonl'))

    def test_completion_exact_set(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): completion_manifest(d,['a','b'],[{'id':'a'}])
            with self.assertRaises(ValueError): completion_manifest(d,['a','a'],[{'id':'a'},{'id':'a'}])
            completion_manifest(d,['a'],[{'id':'a'}])
            self.assertTrue(Path(d,'completion_manifest.json').exists())

    def test_retry_and_persistent_lock_are_bounded(self):
        with patch('engine.result_store.os.replace',side_effect=[PermissionError(),None]) as replace, patch('engine.result_store.time.sleep'):
            replace_with_retry('a','b'); self.assertEqual(replace.call_count,2)
        with patch('engine.result_store.os.replace',side_effect=PermissionError()) as replace, patch('engine.result_store.time.sleep'):
            with self.assertRaises(PermissionError): replace_with_retry('a','b')
            self.assertEqual(replace.call_count,3)

    def test_atomic_failure_preserves_previous_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'progress.json'); atomic_json(p,{'v':1})
            with patch('engine.result_store.os.replace',side_effect=PermissionError()),patch('engine.result_store.time.sleep'):
                with self.assertRaises(PermissionError): atomic_json(p,{'v':2})
            self.assertEqual(json.loads(p.read_text()),{'v':1})
            self.assertEqual(len(list(Path(d).iterdir())),1)
