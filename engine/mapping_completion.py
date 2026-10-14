"""Recover only a truncated final mapping reason; never repair OCR or other fields."""
import json
from copy import deepcopy
from engine.output_contracts import object_schema, validate_shape


def mapping_reason_prefix(raw, schema):
    """Strictly parse complete prefix fields. No invented braces or relaxed JSON.

    Return None for any missing/partial binding, duplicate key, unexpected field,
    complete JSON, or anything except an unterminated final reason string.
    """
    if not isinstance(raw, str):
        return None
    decoder = json.JSONDecoder()
    cursor = 0
    def whitespace(index):
        while index < len(raw) and raw[index].isspace():
            index += 1
        return index
    cursor = whitespace(cursor)
    if raw[cursor:cursor+1] != '{':
        return None
    cursor += 1
    values = {}
    try:
        while True:
            key, cursor = decoder.raw_decode(raw, whitespace(cursor))
            if not isinstance(key, str) or key in values:
                return None
            cursor = whitespace(cursor)
            if raw[cursor:cursor+1] != ':':
                return None
            cursor = whitespace(cursor + 1)
            if key == 'reason':
                if raw[cursor:cursor+1] != '"':
                    return None
                try:
                    decoder.raw_decode(raw, cursor)
                    return None  # Complete reason: missing JSON punctuation is a different failure.
                except json.JSONDecodeError as failure:
                    if not failure.msg.startswith('Unterminated string'):
                        return None
                prefix_schema = object_schema({k: deepcopy(v) for k,v in schema['properties'].items() if k != 'reason'})
                if not validate_shape(values, prefix_schema):
                    return None
                return dict(values, reason='')
            value, cursor = decoder.raw_decode(raw, cursor)
            values[key] = value
            cursor = whitespace(cursor)
            if raw[cursor:cursor+1] != ',':
                return None
            cursor += 1
    except (json.JSONDecodeError, IndexError, TypeError):
        return None
