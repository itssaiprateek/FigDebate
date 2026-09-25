"""Shared audit consistency and opt-in process checks; semantics remain fallible."""
from copy import deepcopy
import hashlib
import json

from engine.output_contracts import object_schema, validate_shape

VERSION = 'process-audit-1'
CHECKS = ('interpretation', 'proposition_alignment', 'condition_coverage', 'relation_inference')


def no_alternative(call):
    """An explicit NONE declaration, not a guess about an uncertain reading.

    Preserve the original wire value. This accepts a lexical spelling of the
    empty alternative only when BOTH typed fields declare its absence.
    """
    text = str(call.get('alternative') or '').strip().casefold()
    return (call.get('alternative_status') == 'NONE'
            and call.get('alternative_relation') == 'UNRESOLVED'
            and (text == '' or text.rstrip('.') == 'none'))


def audit_inconsistency(call, relation):
    """Field incompatibilities only; agreeing labels never establish truth."""
    if not call or call.get('_format_valid') is False:
        return ''
    status = call.get('alternative_status')
    alternative = str(call.get('alternative') or '').strip()
    direction = call.get('alternative_relation')
    if status == 'NONE' and not no_alternative(call):
        return 'NONE conflicts with a supplied alternative or an alternative relation'
    placeholder = alternative.casefold().strip(' .:') in {'', 'none', 'n/a', 'unknown', 'unclear', 'unresolved'}
    if status in {'SAME_DIRECTION', 'DEFEATED', 'UNRESOLVED'} and placeholder:
        return 'The alternative status requires a specific competing reading'
    if status == 'SAME_DIRECTION' and direction != relation:
        return 'SAME_DIRECTION conflicts with the recorded relations'
    if status == 'DEFEATED' and direction == relation:
        return 'DEFEATED records the same direction rather than an opposing reading'
    return ''


def task_record(source):
    return {'version': VERSION, 'source_caption': source,
            'proposition': source,
            'task': 'Assess this source assertion against the image, preserving participants, qualifiers and scope. '
                    'A figurative interpretation is a hypothesis, not a replacement assertion or the speaker attitude.',
            'source_sha256': hashlib.sha256(source.encode()).hexdigest()}


def enabled(runtime):
    return getattr(runtime, 'tribunal_audit_mode', 'baseline') == VERSION


def challenge_schema(schema):
    schema = deepcopy(schema)
    short = {'type': 'string', 'minLength': 1, 'maxLength': 240}
    check = object_schema({'status': {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNRESOLVED']},
        'evidence_ids': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {'type': 'string'}},
        'reason': short})
    schema['properties']['process_checks'] = object_schema({name: deepcopy(check) for name in CHECKS})
    schema['properties']['assessed_interpretation'] = short
    schema['required'] = list(schema['properties'])
    return schema


def process_valid(call, schema, known):
    if call.get('_process_audit_version') != VERSION:
        return False
    try:
        raw = json.loads(call.get('_raw_output', ''))
    except (TypeError, ValueError):
        return False
    expanded = challenge_schema(schema)
    if not validate_shape(raw, expanded):
        return False
    if any(raw.get(k) != call.get(k) for k in expanded['required']):
        return False
    return all(c['status'] == 'PASS' and c['reason'].strip() and set(c['evidence_ids']) <= known
               for c in raw['process_checks'].values())


AUDIT_INSTRUCTIONS = (
    ' Audit the inference step by step in process_checks, citing the selected observations for each check. '
    'State assessed_interpretation as a fallible reading of the complete source. '
    'interpretation: is that reading licensed by the complete expression and image? '
    'proposition_alignment: does the decision assess the source assertion, not substitute speaker attitude? '
    'condition_coverage: were all necessary conditions retained without inventing incidental requirements? '
    'relation_inference: does the relation follow from those facts and that interpretation? '
    'PASS requires a concrete reason; use FAIL or UNRESOLVED honestly. '
    'A same-direction alternative does not repair an unsupported explanation. '
    'A missing fact is material if plausible alternatives could change the source relation while known facts stay fixed. '
    'Inaccessible information can still be necessary. Hypothetical alternatives are not observations. '
    'Do not treat agreeing labels or successful execution as evidence.'
)
