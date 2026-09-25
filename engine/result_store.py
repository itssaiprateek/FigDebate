"""Durable result commits; derived exports may be retried without inference."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def replace_with_retry(source, target, attempts=3):
    if attempts < 1:
        raise ValueError('At least one replacement attempt is required')
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(.1 * (attempt + 1))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def commit_record(directory, record):
    key = record['id']
    path = Path(directory) / 'final_records' / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous.get('sha256') != digest(previous.get('record')):
            raise ValueError('Corrupt committed record')
        if previous['record'] != record:
            raise ValueError('Conflicting final record; use a new run directory')
        return
    atomic_json(path, {'id': key, 'sha256': digest(record), 'record': record})


def read_records(path):
    """Only an unfinished tail may be incomplete; journal and commits must agree."""
    path = Path(path)
    rows = {}
    raw = path.read_bytes() if path.exists() else b''
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines)-1 and not line.endswith(b'\n'):
                break
            raise ValueError('Corrupt complete JSONL record')
        if row['id'] in rows and rows[row['id']] != row:
            raise ValueError('Conflicting duplicate JSONL record')
        rows[row['id']] = row
    for item in sorted((path.parent / 'final_records').glob('*.json')):
        committed = json.loads(item.read_text(encoding='utf-8'))
        row = committed['record']
        if committed.get('id') != row['id'] or committed.get('sha256') != digest(row):
            raise ValueError('Committed result integrity mismatch')
        if row['id'] in rows and rows[row['id']] != row:
            raise ValueError('Journal and committed result disagree')
        rows[row['id']] = row
    return rows


def completion_manifest(directory, selected_ids, records):
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(selected_ids) or len(selected_ids) != len(ids):
        raise ValueError('Completion requires exactly one record per selected ID')
    value = {'status': 'complete', 'samples': len(ids),
             'record_sha256': {r['id']: digest(r) for r in records}}
    atomic_json(Path(directory) / 'completion_manifest.json', value)
    return value
