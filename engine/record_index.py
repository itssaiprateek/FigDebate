"""Disk-backed result mapping; only identities and digests stay in memory."""
from collections.abc import MutableMapping
import hashlib
import json
from pathlib import Path

from engine.result_store import commit_record, digest


class RecordIndex(MutableMapping):
    def __init__(self, directory, resume=False):
        self.directory = Path(directory)
        self.entries = {}
        if resume:
            journal = self.directory / 'records.jsonl'
            if journal.exists():
                with journal.open('rb') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            if not line.endswith(b'\n'):
                                break
                            raise ValueError('Corrupt complete JSONL record')
                        # Migrate legacy journal-only rows into immutable commits.
                        self[row['id']] = row
            for path in sorted((self.directory / 'final_records').glob('*.json')):
                wrapper = json.loads(path.read_text(encoding='utf-8'))
                row = wrapper['record']
                if wrapper.get('id') != row['id'] or wrapper.get('sha256') != digest(row):
                    raise ValueError('Committed result integrity mismatch')
                self._remember(row)

    def _path(self, key):
        return self.directory / 'final_records' / (hashlib.sha256(key.encode()).hexdigest() + '.json')

    def _remember(self, row):
        checksum = digest(row)
        if row['id'] in self.entries and self.entries[row['id']][0] != checksum:
            raise ValueError('Conflicting completed result')
        self.entries[row['id']] = (checksum, row.get('sample', 0))

    def __getitem__(self, key):
        checksum, _ = self.entries[key]
        wrapper = json.loads(self._path(key).read_text(encoding='utf-8'))
        row = wrapper['record']
        if row['id'] != key or wrapper.get('id') != key or wrapper.get('sha256') != checksum or digest(row) != checksum:
            raise ValueError('Committed result integrity mismatch')
        return row

    def __setitem__(self, key, row):
        if row['id'] != key:
            raise ValueError('Result key differs from record identity')
        commit_record(self.directory, row)
        self._remember(row)

    def __delitem__(self, key):
        raise TypeError('Committed results are immutable')

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, key):
        return key in self.entries

    def ordered(self, keys):
        return OrderedRecords(self, sorted(keys, key=lambda key: self.entries[key][1]))


class OrderedRecords:
    sample_ordered = True

    def __init__(self, index, keys):
        self.index, self.keys = index, tuple(keys)

    def __iter__(self):
        return (self.index[key] for key in self.keys)

    def __len__(self):
        return len(self.keys)


def ordered_records(records):
    return records if getattr(records, 'sample_ordered', False) else sorted(records, key=lambda r: r['sample'])
