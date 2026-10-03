"""Bounded distinct-ID grammar for the tribunal's observation-selection wire.

All one-to-four-element subsets remain available, in canonical ID order. This
only constrains identity and ordering; visibility and coverage remain model
judgments. Other schemas use the shared JSON decoder and original validators.
"""
from copy import deepcopy
import json

MAX_CATALOGUE_IDS = 64
_OUTER = ['observations', 'coverage_complete', 'missing_observation']
_ITEM = ['evidence_id', 'supported', 'image_location']


def _exact_object(schema, fields):
    return (isinstance(schema, dict)
            and set(schema) == {'type', 'properties', 'required', 'additionalProperties'}
            and schema['type'] == 'object'
            and isinstance(schema['properties'], dict)
            and list(schema['properties']) == fields
            and schema['required'] == fields
            and schema['additionalProperties'] is False)


def eligible(schema):
    """Recognize only the tested wire; never discard another schema's limits."""
    if not _exact_object(schema, _OUTER):
        return False
    props = schema['properties']
    array = props['observations']
    if (not isinstance(array, dict)
            or set(array) != {'type', 'minItems', 'maxItems', 'items'}
            or array['type'] != 'array'
            or array['minItems'] != 1 or array['maxItems'] != 4
            or not _exact_object(array['items'], _ITEM)):
        return False
    item = array['items']['properties']
    identity = item['evidence_id']
    if (not isinstance(identity, dict) or set(identity) != {'type', 'enum'}
            or identity['type'] != 'string'
            or item['supported'] != {'type': 'boolean'}
            or props['coverage_complete'] != {'type': 'boolean'}):
        return False
    ids = identity['enum']
    return (isinstance(ids, list) and 0 < len(ids) <= MAX_CATALOGUE_IDS
            and all(isinstance(identifier, str) and identifier for identifier in ids)
            and len(ids) == len(set(ids)))


def grammar(schema, whitespace_limit):
    """Factor suffix choices to avoid combinatorial enumeration of ID tuples."""
    if not eligible(schema):
        raise ValueError('Unsupported distinct-selection schema')
    props = schema['properties']
    item = props['observations']['items']
    ids = sorted(item['properties']['evidence_id']['enum'])
    parts = ['{', '"observations"', ':', '[', 'selections', ']', ',',
             '"coverage_complete"', ':', 'flag', ',', '"missing_observation"',
             ':', 'missing', '}']
    body = ' ws '.join(p if p in {'selections', 'flag', 'missing'} else json.dumps(p)
                       for p in parts)
    rules = ['start: ws body ws', 'body: ' + body,
             'ws: /[ \\t\\r\\n]{0,' + str(whitespace_limit) + '}/',
             'selections: s_0_4']

    def json_rule(name, definition):
        definition = deepcopy(definition)
        definition['x-guidance'] = {
            **definition.get('x-guidance', {}),
            'whitespace_pattern': rf'[\x20\x0A\x0D\x09]{{1,{whitespace_limit}}}'}
        return name + ': %json ' + json.dumps(definition)

    rules.extend([json_rule('flag', props['coverage_complete']),
                  json_rule('missing', props['missing_observation'])])
    for index, identifier in enumerate(ids):
        entry = deepcopy(item)
        entry['properties']['evidence_id']['enum'] = [identifier]
        rules.append(json_rule(f'o_{index}', entry))
        for remaining in range(1, 5):
            options = [f'o_{index}']
            if index + 1 < len(ids):
                if remaining > 1:
                    options[0] += f' (ws "," ws s_{index + 1}_{remaining - 1})?'
                options.append(f's_{index + 1}_{remaining}')
            rules.append(f's_{index}_{remaining}: ' + ' | '.join(options))
    return '\n'.join(rules)
