"""Independent evidence selection and a source-bound semantic dispute check.

The model still supplies semantic judgments. Structural checks are necessary,
not proof that those judgments are true. No labels or sample identities enter.
"""
from copy import deepcopy
import hashlib
import json

from engine import evidence_verification as core
from engine.output_contracts import object_schema, validate_shape
from engine.tribunal_protocol import SHORT, PROSE, IDS, LITERAL_TYPES

VERSION = '5.0'
PROTOCOL = 'evidence-review-5.0'


def token_bounded_text_schema(schema):
    """Keep the finite call budget; do not force strings closed mid-clause.

    Source quotations are exact spans and may also exceed the old character
    limit. Array/cardinality, citation, source-binding and completion checks stay.
    """
    schema = deepcopy(schema)
    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get('type') == 'string' and 'enum' not in node:
            node.pop('maxLength', None)
        for child in node.get('properties', {}).values():
            visit(child)
        visit(node.get('items'))
    visit(schema)
    return schema


DECISION = token_bounded_text_schema(core.DECISION)
SELECTION = object_schema({**deepcopy(core.VISUAL['properties']),
    'coverage_complete': {'type': 'boolean'}, 'missing_observation': SHORT})
LEGACY_SELECTION_WIRE = object_schema({
    'observations': {'type': 'array', 'minItems': 1, 'maxItems': 4,
        'items': object_schema({'evidence_id': {'type': 'string'},
                               'supported': {'type': 'boolean'}, 'attachment': SHORT})},
    'coverage_complete': {'type': 'boolean'}, 'missing_observation': SHORT})
LOCATION_PROTOCOL = 'image_location_only_v1'
SELECTION_WIRE = deepcopy(LEGACY_SELECTION_WIRE)
_location_item = SELECTION_WIRE['properties']['observations']['items']
_location_item['properties']['image_location'] = _location_item['properties'].pop('attachment')
_location_item['required'] = list(_location_item['properties'])
VISUAL_SELECTION_INSTRUCTIONS = (
    'Inspect the FULL image and source caption. Select one to four deciding factual observations '
    'from the catalogue; select independently, without a proposed answer. Include relevant opposing '
    'evidence, both comparison sides and relevant panels/times. Prefer the smallest sufficient set. '
    'For each selected ID verify its COMPLETE record text and image location against pixels. supported=false if '
    'that record is inaccurate. Copying image text verifies visibility, not its real-world truth. '
    'Do not replace an observation with the caption or an inferred meaning. coverage_complete means '
    'the selected observations cover the relevant visible facts, not that the caption is true. '
    'If an essential visible fact is missing from the catalogue or unclear, set coverage_complete=false '
    'and missing_observation to ONE observable question for the visual witness; otherwise use an empty string. '
    'Do not demand hidden intentions or literal events for an idiom. Return IDs, supported flags and '
    'brief image_location values only; software copies their source text. Do not generate observations or a verdict. '
    'image_location names ONLY WHERE the record is pictured: its panel, region, depicted object or speaker. '
    'Use a short location phrase. Do not transcribe text or describe actions in image_location. ')
# Examine the alternative before certifying the proposed argument.
CHALLENGE = object_schema({key: deepcopy(core.CHALLENGE['properties'][key]) for key in (
    'alternative', 'alternative_relation', 'alternative_status', 'deciding_evidence_ids',
    'decision_errors', 'role_scope_errors', 'reason')})
# Text is bounded by the call's output/context/time budget, not by a second
# character ceiling that can force a valid JSON string to end mid-sentence.
for _field in ('alternative', 'reason'):
    CHALLENGE['properties'][_field].pop('maxLength', None)
for _field in ('decision_errors', 'role_scope_errors'):
    CHALLENGE['properties'][_field]['items'].pop('maxLength', None)

CHALLENGE_REPAIR_FIELDS = ('alternative', 'alternative_relation', 'alternative_status',
                           'deciding_evidence_ids', 'reason')

CHALLENGE_INSTRUCTIONS = (
    'Audit the candidate against the FULL image and source caption. The source proposition cannot change. '
    'Check that interpreted_assertion preserves it and draft_argument addresses it with grounded evidence. '
    'Only report errors that actually invalidate a candidate inference: name that inference and its counterevidence. '
    'Do not list an objection that your own explanation resolves or says does not invalidate the decision. '
    'A false caption can correctly receive CONFLICT. Missing support alone is not counterevidence. '
    'Consider a plausible alternative only if grounded in these sources; do not invent one to fill a field. '
    'Same relation means SAME_DIRECTION, not DEFEATED or an unresolved decision. '
    'Opposing reading: DEFEATED requires cited distinguishing evidence; otherwise UNRESOLVED. '
    'No alternative: empty alternative, UNRESOLVED relation, NONE status. '
    'Use empty error lists when no material error remains. Cite deciding evidence even for NONE. '
    'Each text field is one short complete sentence.'
)


def challenge_contract(value, relation):
    from engine.tribunal_process import audit_inconsistency
    error = audit_inconsistency(value, relation)
    if error:
        return {'kind': 'AUDIT_INCOMPLETE', 'message': error}


def condition_assessments_consistent(value):
    """A condition has one state; this check does not establish semantic truth."""
    unresolved = {core.norm(x) for x in value.get('unestablished_conditions', [])}
    states = {}
    for item in value.get('condition_checks', []):
        quote, relation = core.norm(item['caption_quote']), item['relation']
        if (quote in unresolved and relation != 'UNRESOLVED') or (quote in states and states[quote] != relation):
            return False
        states[quote] = relation
    return True


def complete_audit_response(call):
    """Check full JSON or a recorded, exactly reconstructed field clarification.

    A failed semantic clarification may retain its patch as raw output. This is
    not invalid JSON, and must not be mistaken for execution failure. Never use
    this function to certify the audit's meaning or to authorize acceptance.
    """
    from engine.tribunal_process import challenge_schema
    schema = challenge_schema(CHALLENGE) if call.get('_process_audit_version') else CHALLENGE
    try:
        raw = json.loads(call.get('_raw_output', ''))
    except (TypeError, ValueError):
        return False
    projected = {k: call.get(k) for k in schema['required']}
    if validate_shape(raw, schema):
        return raw == projected
    repair = call.get('_field_repair_audit') or {}
    paths = repair.get('paths') or {}
    if (not isinstance(raw, dict) or set(raw) != set(CHALLENGE_REPAIR_FIELDS)
            or paths != {k: [k] for k in CHALLENGE_REPAIR_FIELDS}
            or not validate_shape(repair.get('original'), schema)):
        return False
    rebuilt = dict(repair['original'], **raw)
    return validate_shape(rebuilt, schema) and rebuilt == projected


def catalog(ledger):
    """Deterministic literal catalogue, deduplicated without merging attachments."""
    from engine.semantic_bridge import VISUAL_SOURCES
    from engine.evidence_ledger import is_admissible_evidence
    eligible = [x for x in ledger if x.get('id') and x.get('grounded')
                and x.get('source') in VISUAL_SOURCES and x.get('type') in LITERAL_TYPES
                and is_admissible_evidence(ledger, x)]
    result, seen = [], set()
    for x in sorted(eligible, key=lambda item: str(item['id'])):
        # Preserve differences in panel/region/speaker and provenance. Exact
        # copied records may share an alias; distinct observations stay distinct.
        attachment = {k: x.get(k) for k in ('panel', 'region', 'speaker', 'attachment', 'derived_from_ids')}
        key = json.dumps([core.norm(x.get('text')), attachment], sort_keys=True)
        if key not in seen:
            result.append(x)
            seen.add(key)
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def binding(proposal, ledger):
    """Bind ALL eligible records, not only the proposer's chosen fragment."""
    from engine.independent_review import verification_subject
    subject = verification_subject(proposal, ledger)
    subject['observations'] = sorted(subject['observations'], key=lambda x: x['id'])
    provenance = [{k: x.get(k) for k in ('id', 'source', 'type', 'grounded', 'lifecycle_status',
        'text', 'derived_from_ids', 'panel', 'region', 'speaker', 'attachment')}
        for x in catalog(ledger)]
    data = [VERSION, subject, proposal.get('proposed_relation'),
                    proposal.get('bridge_statement'), proposal.get('counter_interpretation'),
                    proposal.get('verification_task', {}), provenance,
                    proposal.get('interpreted_assertion', '')]
    if proposal.get('process_audit_version'):
        data.append(proposal['process_audit_version'])
    return _digest(data)


def selection_valid(value, records):
    by_id = {x['id']: x for x in records}
    observations = value.get('observations') or []
    selected = [x.get('evidence_id') for x in observations]
    try:
        wire = json.loads(value.get('_raw_output', ''))
    except (TypeError, ValueError):
        return False
    projected = {'observations': [{k: x.get(k) for k in ('evidence_id', 'supported', 'attachment')}
                                  for x in observations],
                 'coverage_complete': value.get('coverage_complete'),
                 'missing_observation': value.get('missing_observation')}
    schema = LEGACY_SELECTION_WIRE
    if value.get('_location_wire_protocol'):
        if value['_location_wire_protocol'] != LOCATION_PROTOCOL:
            return False
        schema = SELECTION_WIRE
        for observation in projected['observations']:
            observation['image_location'] = observation.pop('attachment')
    return bool(core.executed(value) and validate_shape(wire, schema) and wire == projected
                and not core.capped_generated_clause(wire, schema)
                and not core.unfinished_generated_field(projected) and selected
                and len(selected) == len(set(selected)) and set(selected) <= set(by_id)
                and value.get('coverage_complete') is True and not value.get('missing_observation', '').strip()
                and all(x.get('supported') is True and x.get('observed', '').strip()
                        and x['observed'] == by_id[x['evidence_id']].get('text', '')
                        and x.get('attachment', '').strip() for x in observations))


def challenge_valid(value, known, relation):
    # Even NONE must identify the factual basis for dismissing alternatives.
    from engine.tribunal_process import audit_inconsistency
    return bool(core.challenge_valid(value, known, relation, CHALLENGE)
                and complete_audit_response(value)
                and not audit_inconsistency(value, relation)
                and value.get('deciding_evidence_ids') and value.get('reason', '').strip())


def decision_valid(value, known, source):
    """Validate the exact generated judgment, including non-indexed transports."""
    if value.get('_semantic_contract_valid') is False or not core.decision_valid(value, known, source, DECISION):
        return False
    # Indexed responses were already reconstructed and compared by the core
    # validator. Plain JSON must satisfy the same raw-to-record identity rule.
    if value.get('_source_span_protocol') == 'indexed_source_tokens_v1':
        return True
    try:
        raw = json.loads(value.get('_raw_output', ''))
    except (TypeError, ValueError):
        return False
    return bool(validate_shape(raw, DECISION)
                and all(raw[k] == value.get(k) for k in DECISION['required']))


def citation_schema(schema, known):
    """Constrain transport identities without constraining semantic judgments."""
    schema = deepcopy(schema)
    def visit(node):
        if not isinstance(node, dict):
            return
        for key, child in node.get('properties', {}).items():
            if key in {'evidence_ids', 'deciding_evidence_ids'}:
                child['items'] = {'type': 'string', 'enum': sorted(known)}
                child['minItems'] = 1
            visit(child)
        visit(node.get('items'))
    visit(schema)
    return schema


def mapping_schema(known):
    schema = citation_schema(token_bounded_text_schema(core.MAPPING), known)
    # An entirely unmatched caption must not require an invented binding.
    schema['properties']['bindings']['minItems'] = 0
    return schema


def mapping_error(value, source, known):
    if not value['bindings'] and not any(x.strip() for x in value['unmatched_roles']):
        return 'Provide a supported binding or explicitly name an unmatched role'
    for item in value['bindings']:
        if not core.quotation(item['caption_quote'], source):
            return core.source_quote_error(item['caption_quote'], source)
        if not item['evidence_ids'] or not set(item['evidence_ids']) <= known:
            return 'Bind only selected observation IDs'


def mapping_response_valid(call, known, source):
    """An honest unmatched response is valid execution, not a usable mapping.

    Recheck the actual wire contract and source binding; never infer validity
    from a saved ``verified`` flag. Acceptance still requires mapping_valid.
    """
    schema = mapping_schema(known)
    if not core.payload_valid(call, schema) or not core.source_span_record_valid(call, schema, source):
        return False
    if call.get('_source_span_protocol') != 'indexed_source_tokens_v1':
        try:
            raw = json.loads(call.get('_raw_output', ''))
        except (ValueError, TypeError):
            return False
        if not validate_shape(raw, schema):
            return False
        rebound = core.bind_mapping_source_spans(raw, source)
        if (any(rebound.get(key) != call.get(key) for key in schema['required'])
                or rebound.get('_source_quote_bindings') != call.get('_source_quote_bindings')):
            return False
    return bool(not mapping_error(call, source, known)
                and not core.capped_generated_clause(call, schema)
                and all(x.strip() for x in call['unmatched_roles'])
                and all(b['observed_entity'].strip() and b['role_scope'].strip() for b in call['bindings']))


def mapping_valid(call, known, source):
    """A successful mapping additionally resolves every reported role."""
    return bool(mapping_response_valid(call, known, source)
                and call['bindings'] and not call['unmatched_roles'])


def verify(runtime, image, proposal, ledger):
    from engine.independent_review import image_subject_hash
    from engine import tribunal_process as process
    process_mode = proposal.get('process_audit_version') == process.VERSION
    source = proposal.get('source_caption', '')
    records = catalog(ledger)
    available = {x['id'] for x in records}
    record = {'schema_version': VERSION, 'provenance_sha256': binding(proposal, ledger),
        'observed_image_sha256': image_subject_hash(image), 'calls': [], 'obligations': {},
        'method': 'independent_evidence_selection_blind_relation_discriminating_challenge',
        'model_error_independence_established': False}
    if process_mode:
        record['process_audit_version'] = process.VERSION

    def finish(reason=None):
        if reason:
            record['stopped_after'] = reason
        calls = list(record['obligations'].values()) + record['calls']
        record['generation_call_count'] = sum('_execution_status' in x and not x.get('_cache_hit') for x in calls)
        record['_generation_seconds'] = sum(x.get('_generation_seconds', 0) for x in calls)
        return record

    if (not source or not available or not record['observed_image_sha256']
            or record['observed_image_sha256'] != proposal.get('image_sha256')):
        record['input_binding_error'] = 'missing_or_mismatched_case_inputs'
        return finish('case_binding')
    if core.norm(source) != core.norm(proposal.get('caption_premise')):
        record['input_binding_error'] = 'caption_must_preserve_exact_source'
        return finish('caption_preservation')
    record['obligations']['caption'] = {'verified': True, 'method': 'exact_source_identity'}
    # All downstream caches also depend on the complete candidate catalogue.
    generate = core.obligation_runner(runtime, image, source, records, available, VERSION)
    task = proposal.get('verification_task') or {}
    def ask(name, instructions, payload, *args, **kwargs):
        if process_mode:
            payload = dict(payload, task=process.task_record(source))
        stages = {'role_scope_mapping': {'mapping', 'relation', 'challenge'},
                  'contradictory_audit': {'challenge'},
                  'audit_objection': {'challenge'},
                  'unresolved_relation': {'relation', 'challenge'},
                  'disputed_inference': {'relation', 'challenge'},
                  'relation_disagreement': {'relation', 'challenge'}}
        if name in stages.get(task.get('failed_requirement'), set()):
            payload = dict(payload, targeted_check={k: task.get(k, '') for k in
                ('failed_requirement', 'question', 'disputed_detail', 'instruction')})
            instructions += ' The targeted_check is a fallible question, not evidence or a supplied verdict. Resolve it from the original sources.'
        return generate(name, instructions, payload, *args, **kwargs)
    schema = deepcopy(SELECTION_WIRE)
    schema['properties']['observations']['items']['properties']['evidence_id'] = {
        'type': 'string', 'enum': sorted(available)}
    visual = ask('visual', VISUAL_SELECTION_INSTRUCTIONS,
        {'source_caption': source, 'observations': [{'id': x['id'], 'text': x.get('text', '')} for x in records]},
        schema, 256,
        validator=lambda v: None if len({x['evidence_id'] for x in v['observations']}) == len(v['observations'])
        and all(x['evidence_id'] in available for x in v['observations']) else 'Select distinct catalogue IDs',
        repair_field='observations')
    record['obligations']['visual'] = visual
    for observation in visual.get('observations') or []:
        # Field translation only: preserve exactly what the model said and
        # keep the original wire output for independent validation below.
        if 'image_location' in observation:
            observation['attachment'] = observation.pop('image_location')
        observation['observed'] = next((x.get('text', '') for x in records if x['id'] == observation['evidence_id']), '')
    visual['_location_wire_protocol'] = LOCATION_PROTOCOL
    visual['_observation_text_binding'] = 'exact_catalogue_record_by_id_v1'
    visual['verified'] = selection_valid(visual, records)
    if not visual['verified']:
        return finish('visual_grounding')
    known = {x['evidence_id'] for x in visual['observations']}
    record['selected_evidence_ids'] = sorted(known)
    observations = [{k: v for k, v in x.items() if k != 'supported'} for x in visual['observations']]
    mapping = ask('mapping',
        'Bind the caption participants to the observed participants. Give one concise binding per necessary '
        'role/scope, citing selected IDs. Quote the source caption exactly. Match subject, object, speaker, '
        'One binding may cite multiple observations. Do not create a binding for each observation or incidental object. '
        'panel and time where relevant. Matching identifies WHAT is compared, not whether its state agrees. '
        'An intended outcome is not an actual outcome. An author reaction is not the utility or belief of '
        'a depicted character. A licensed metaphor may map corresponding roles without literal participants. '
        'Use unmatched_roles only for genuinely ambiguous identity/scope; opposing properties still match.',
        {'source_caption': source, 'observations': observations}, mapping_schema(known), 320,
        validator=lambda v: mapping_error(v, source, known),
        picture=None, source_binder=lambda v: core.bind_mapping_source_spans(v, source))
    record['obligations']['mapping'] = mapping
    mapping['response_valid'] = mapping_response_valid(mapping, known, source)
    mapping['verified'] = mapping_valid(mapping, known, source)
    if not mapping['verified']:
        return finish('entity_scope_mapping')
    case = {'source_caption': source, 'observations': observations, 'bindings': mapping['bindings']}
    # The enclosing expression is immutable context, not a model-generated
    # quotation or an inferred premise. All modes need it, including baseline.
    case['complete_source_expression'] = source
    if process_mode:
        # Carry a complete enclosing expression; never silently expand a model quote.
        case['complete_source_expressions'] = [
            {'focused_quote': item['caption_quote'], 'complete_expression': source,
             'role_scope': item['role_scope']} for item in mapping['bindings']]

    def decision_contract(value):
        quotes = [x['caption_quote'] for x in value['condition_checks']] + value['unestablished_conditions']
        for quote in quotes:
            if not core.quotation(quote, source):
                return core.source_quote_error(quote, source)
        if not value['evidence_ids'] or not set(value['evidence_ids']) <= known:
            return 'Cite selected observation IDs only'
        relations = [x['relation'] for x in value['condition_checks']]
        if not condition_assessments_consistent(value):
            return {'kind': 'SEMANTIC_INCONSISTENCY',
                    'message': 'The same condition cannot be established or contradicted and also unresolved. Reconcile its assessment.'}
        if value['relation'] == 'SUPPORT' and (value['unestablished_conditions'] or any(x != 'SUPPORT' for x in relations)):
            return {'kind': 'SEMANTIC_INCONSISTENCY',
                    'message': 'SUPPORT requires all necessary conditions; retain unresolved conditions honestly'}
        if value['relation'] == 'CONFLICT' and 'CONFLICT' not in relations:
            return {'kind': 'SEMANTIC_INCONSISTENCY', 'message': 'CONFLICT requires a positively incompatible condition'}

    decision = ask('relation',
        'Decide the source-caption relation afresh from these observations and the full image. '
        'Compare the same subject, property and scope. In condition_checks quote necessary source '
        'conditions and give the actual image state. Evaluate both sides of comparisons and actual '
        'outcomes separately from intentions. SUPPORT establishes the full claim including qualifiers. '
        'CONFLICT requires a positive incompatibility. The absence of printed sentiment, a literal '
        'idiom event or unavailable history is not a conflicting state. Missing necessary evidence '
        'is UNRESOLVED. A depicted trigger can ground an evaluative reaction. Do not substitute a '
        'different assertion, silently reverse sarcasm, or remove qualifiers to obtain an answer.'
        ' Failure to observe a property does not prove its opposite. Do not turn unknown intentions '
        'into observed facts. State which positive observation is incompatible with a CONFLICT condition. '
        'Grounded figurative inference is allowed; literal depiction of an idiom is not required.'
        + (' Check the complete_source_expressions, not only focused quotes. Retain negation, comparisons, '
           'time and modality. Do not add historical requirements unless the decision depends on them.' if process_mode else ''),
        case, citation_schema(DECISION, known), 384, validator=decision_contract)
    if decision.get('condition_checks'):
        decisive = next((x for x in decision['condition_checks'] if x['relation'] == decision.get('relation')),
                        decision['condition_checks'][0])
        decision.update(decisive_caption_quote=decisive['caption_quote'], decisive_observation=decisive['image_state'])
    record['calls'].append(decision)
    if not decision_valid(decision, known, source) or decision.get('relation') not in {'SUPPORT', 'CONFLICT'}:
        return finish('relation_direction')
    challenge = ask('challenge',
        CHALLENGE_INSTRUCTIONS
        + (process.AUDIT_INSTRUCTIONS if process_mode else ''),
        dict(case, decision={k: decision[k] for k in core.DECISION['required']},
             interpreted_assertion=proposal.get('interpreted_assertion', ''),
             draft_argument=proposal.get('bridge_statement', ''), draft_visual_premise=proposal.get('visual_premise', '')),
        citation_schema(process.challenge_schema(CHALLENGE) if process_mode else CHALLENGE, known),
        768 if process_mode else 512,
        validator=lambda v: challenge_contract(v, decision.get('relation')),
        repair_fields=CHALLENGE_REPAIR_FIELDS)
    if process_mode:
        challenge['_process_audit_version'] = process.VERSION
    record['obligations']['arguments'] = challenge
    return finish()


def audit(proposal, ledger):
    record = proposal.get('independent_verification') or {}
    obligations = record.get('obligations') or {}
    visual = obligations.get('visual') or {}
    records = catalog(ledger)
    known = set(record.get('selected_evidence_ids') or [])
    source, relation = proposal.get('source_caption', ''), proposal.get('proposed_relation')
    bound = bool(record.get('schema_version') == VERSION and source and proposal.get('image_sha256')
        and record.get('observed_image_sha256') == proposal['image_sha256']
        and record.get('provenance_sha256') == binding(proposal, ledger))
    caption = bool(source and core.norm(source) == core.norm(proposal.get('caption_premise')))
    visual_ok = selection_valid(visual, records) and known == {x['evidence_id'] for x in visual.get('observations', [])}
    mapping_call = obligations.get('mapping', {})
    mapping_ok = mapping_valid(mapping_call, known, source)
    calls = record.get('calls') or []
    decision = calls[0] if len(calls) == 1 else {}
    decision_ok = decision_valid(decision, known, source) and condition_assessments_consistent(decision)
    agrees = decision_ok and relation in {'SUPPORT', 'CONFLICT'} and decision.get('relation') == relation
    args = obligations.get('arguments', {})
    counter = challenge_valid(args, known, relation)
    if proposal.get('process_audit_version') or record.get('process_audit_version') or args.get('process_checks'):
        from engine import tribunal_process as process
        counter = bool(counter and proposal.get('process_audit_version') == record.get('process_audit_version') == process.VERSION
                       and process.process_valid(args, CHALLENGE, known))
    roots = [name for name, passed in (('case_binding', bound), ('caption_preservation', caption),
        ('visual_grounding', visual_ok), ('entity_scope_mapping', mapping_ok),
        ('relation_direction', agrees), ('material_counterargument', counter)) if not passed]
    return {'valid': not roots, 'bound_to_current_case': bound, 'executed': decision_ok,
        'caption_verified': caption, 'visual_verified': visual_ok, 'entity_scope_verified': mapping_ok,
        'mapping_response_valid': mapping_response_valid(mapping_call, known, source),
        'premises_verified': bool(bound and caption and visual_ok and mapping_ok),
        'relation_agreement': bool(agrees), 'arguments_verified': bool(counter and agrees),
        'counter_resolved': counter, 'actual_order_swap': False,
        'verification_control': 'independent_selection_and_discriminating_challenge',
        'root_failures': roots, 'failed_obligations': roots,
        'counter_resolution_basis': args.get('alternative_status', 'NOT_EXECUTED')}
