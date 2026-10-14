"""Experimental full-claim condition assessment; consistency is not truth.

Until decomposition is independently qualified, C1 is the COMPLETE source.
No model-selected substring is allowed to replace that obligation. The model
supplies evidence and semantics; software derives only redundant statuses.
"""
from copy import deepcopy
import hashlib
import json
from engine.output_contracts import object_schema, validate_shape, incomplete_clause
from engine.semantic_protocol import READING, reading_error

VERSION = 'full-claim-conditions-1'
CANONICAL_VERSION = 'canonical-claim-conditions-1'
TEXT = {'type': 'string', 'minLength': 1}
INSTRUCTIONS = (
    'Assess the immutable obligation C1: the ENTIRE source caption, not a selected fragment. '
    'Write evidence before assessment. state_pairs compare actual and asserted states for the SAME scope, distinguishing '
    'each speaker, subject, comparison side and time. A wish, prediction, instruction or quoted '
    'character belief is not an actual outcome or the author\'s belief. '
    'connection explains observable basis -> licensed figurative connection -> asserted property; '
    'for literal claims compare directly. Topic overlap alone is insufficient. Preserve negation, '
    'modality and ALL material qualifiers. A metaphor does not require its source object to exist. '
    'coverage is COMPLETE only if all decision-relevant parts have been assessed; PARTIAL or UNCLEAR '
    'means that the full claim remains unresolved. assessment is SUPPORT when the whole claim is '
    'supported, CONFLICT for positive incompatibility with the whole claim, otherwise UNRESOLVED. '
    'A false conjunct can disprove a conjunction; a false alternative alone cannot disprove a disjunction. '
    'Use one state_pair for a single subject and two for a comparison or sequence. scope identifies participant, speaker and time. '
    'Do not infer opposition from missing evidence. Keep each state a short phrase and connection one short '
    'complete sentence each. reason is one complete explanation of the whole-claim decision. '
)


def task(source):
    return {'protocol': VERSION, 'claim_id': 'C1', 'source_caption': source,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'representation': 'UNDECOMPOSED_SOURCE', 'decomposition_semantically_verified': False,
            'interpretation_authority': 'hypothesis_only_preserve_asserted_claim'}


def schema(known, canonical=False):
    result = object_schema({'reading': deepcopy(READING), 'conditions': {
        'type': 'array', 'minItems': 1, 'maxItems': 1, 'items': object_schema({
            'claim_id': {'type': 'string', 'enum': ['C1']},
            'evidence_ids': {'type': 'array', 'minItems': 1, 'maxItems': 4,
                             'items': {'type': 'string', 'enum': sorted(known)}},
            'state_pairs': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                'items': object_schema({'scope': TEXT, 'actual': TEXT, 'asserted': TEXT})},
            'connection': TEXT,
            'coverage': {'type': 'string', 'enum': ['COMPLETE', 'PARTIAL', 'UNCLEAR']},
            'assessment': {'type': 'string', 'enum': ['SUPPORT', 'CONFLICT', 'UNRESOLVED']},
        })}, 'reason': TEXT})
    if canonical:
        row = result['properties']['conditions']['items']
        row['properties'].pop('state_pairs')
        row['properties'].pop('connection')
        row['properties']['image_state'] = TEXT
        row['required'] = list(row['properties'])
    return result


CANONICAL_INSTRUCTIONS = (
    'Assess C1, the COMPLETE original caption, preserving all roles, polarity, modality and qualifiers. '
    'In image_state state the deciding observation and its participant/panel/time attachment. '
    'Do not treat a wish or quoted belief as an actual outcome. Explain the licensed image-to-claim '
    'connection in reason; for figurative claims a literal source object is not necessary. '
    'coverage is COMPLETE only if the necessary scope was assessed; otherwise PARTIAL or UNCLEAR. '
    'assessment is SUPPORT for a supported full claim, CONFLICT for positive incompatibility with '
    'a necessary part, or UNRESOLVED for missing necessary evidence. A false alternative alone '
    'does not refute a disjunction. Give one complete reason, including any material uncertainty. '
    'Software derives the overall relation and unresolved fields from this single assessment. '
)


def error(value):
    for row in value['conditions']:
        if len(row['evidence_ids']) != len(set(row['evidence_ids'])):
            return 'Cite distinct selected observations.'
        if 'image_state' in row:
            if not row['image_state'].strip() or incomplete_clause(row['image_state'], conjunctions=True):
                return 'Finish the deciding observation.'
            continue
        if (incomplete_clause(row['connection'], conjunctions=True) or any(
                not item[k].strip() or incomplete_clause(item[k], conjunctions=True)
                for item in row['state_pairs'] for k in ('scope', 'actual', 'asserted'))):
            return 'Finish the observation and connection without changing their meaning.'
    if incomplete_clause(value['reason'], conjunctions=True):
        return 'Finish the full-claim explanation.'
    row = value['conditions'][0]
    relation = row['assessment'] if row['coverage'] == 'COMPLETE' else 'UNRESOLVED'
    invalid = reading_error(dict(value, relation=relation))
    if invalid:
        return {'kind': 'SEMANTIC_INCONSISTENCY', 'message': invalid}


def project(value, source):
    row = value['conditions'][0]
    relation = row['assessment'] if row['coverage'] == 'COMPLETE' else 'UNRESOLVED'
    return {'reading': deepcopy(value['reading']), 'reason': value['reason'],
            'relation': relation, 'evidence_ids': list(row['evidence_ids']),
            'condition_checks': [{'caption_quote': source,
                'image_state': row['image_state'] if 'image_state' in row else '; '.join(f"{x['scope']}: {x['actual']}" for x in row['state_pairs']), 'relation': relation}],
            'unestablished_conditions': [source] if relation == 'UNRESOLVED' else []}


def attach(call, source, known, canonical=False):
    shape = schema(known, canonical)
    value = {key: deepcopy(call.get(key)) for key in shape['required']}
    if not validate_shape(value, shape):
        return call
    call.update(project(value, source))
    call['_condition_protocol'] = CANONICAL_VERSION if canonical else VERSION
    call['_condition_value'] = value
    call['_claim_target'] = task(source)
    return call


def valid(call, known, source):
    from engine.evidence_verification import executed
    canonical = call.get('_condition_protocol') == CANONICAL_VERSION
    if (call.get('_condition_protocol') not in {VERSION, CANONICAL_VERSION} or not executed(call)
            or call.get('_semantic_contract_valid') is False or call.get('_claim_target') != task(source)):
        return False
    try:
        wire = json.loads(call.get('_raw_output', ''))
    except (ValueError, TypeError):
        return False
    if not validate_shape(wire, schema(known, canonical)) or error(wire):
        return False
    if wire != call.get('_condition_value') or wire['conditions'] != call.get('conditions'):
        return False
    return all(call.get(k) == v for k, v in project(wire, source).items())
