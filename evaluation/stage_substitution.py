"""Evaluation-only stage substitution. Never imported by production inference.

A checked visual selection, claim-role mapping or condition assessment may be
substituted to measure downstream sensitivity. These are ORACLE diagnostics,
not accuracy improvements. The original source caption is never replaced.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from engine.output_contracts import validate_shape

SIGNATURES = {'visual': {'observations', 'coverage_complete'},
              'claim_roles': {'bindings', 'unmatched_roles'},
              'conditions': {'conditions', 'reading'}}


@contextmanager
def substitute_stage(runtime, entry, source):
    result = {'oracle': bool(entry), 'substitutions': 0, 'stage': None}
    if not entry:
        yield result
        return
    if set(entry) != {'source_sha256','stage','value','reviewed_by','basis'}:
        raise ValueError('Oracle entry requires exact source identity, stage, value and reviewer provenance')
    if (entry['source_sha256'] != hashlib.sha256(source.encode()).hexdigest()
            or entry['stage'] not in SIGNATURES or not entry['reviewed_by'].strip() or not entry['basis'].strip()):
        raise ValueError('Unreviewed or mismatched oracle intervention')
    original = runtime.generate
    signature = SIGNATURES[entry['stage']]
    result.update(stage=entry['stage'], entry_sha256=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest())
    def generate(image, prompt, max_new_tokens=None, json_schema=None):
        if signature <= set((json_schema or {}).get('properties', {})):
            if result['substitutions'] or not validate_shape(entry['value'], json_schema):
                raise ValueError('Oracle substitution failed the actual stage contract')
            result['substitutions'] += 1
            runtime._last_generation_diagnostics = {'oracle_stage_substitution': entry['stage'], 'generated_tokens': 0}
            return json.dumps(deepcopy(entry['value'])), 0.0
        return original(image, prompt, max_new_tokens=max_new_tokens, json_schema=json_schema)
    runtime.generate = generate
    try:
        yield result
    finally:
        runtime.generate = original


def load_interventions(path, identifiers):
    if path is None:
        return {}, None
    raw = path.read_bytes()
    entries = json.loads(raw)
    if not isinstance(entries, dict) or set(entries) != set(identifiers):
        raise ValueError('Exactly one intervention per selected case is required')
    return entries, hashlib.sha256(raw).hexdigest()
