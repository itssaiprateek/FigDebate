"""One diagnostic-driven repair hearing; feedback cannot authorize a label."""
from engine.review_routing import route_question, route_followup, new_semantic_questions

# Criticism is not new evidence. Recheck the disputed proof of this candidate;
# only a witness hearing with new evidence may require a new proposal.
FROZEN_REPAIR_ISSUES = frozenset({'contradictory_audit', 'audit_objection',
    'relation_disagreement', 'unresolved_relation', 'disputed_inference'})


def validate_options(mode, audit_mode, judge_mode, debate_mode, protocol, feedback_mode):
    if mode not in {'disabled', 'bounded'} or audit_mode not in {'baseline', 'process-audit-1'}:
        raise ValueError('Unknown tribunal repair/audit option')
    if mode != 'disabled' or audit_mode != 'baseline':
        if judge_mode != 'tribunal' or debate_mode != 'enabled' or protocol != 'evidence-review-5.0':
            raise ValueError('Experimental tribunal options require tribunal, enabled hearings and the explicit review5 profile')
        if feedback_mode != 'disabled':
            raise ValueError('Tribunal-only options require feedback disabled for attribution')


def scheduled_repair(review, debate, resolution, *, mode='disabled', feedback_mode='disabled',
                     round_number=1, max_rounds=2):
    if (round_number >= min(max_rounds, 2) or resolution.get('accepted')
            or resolution.get('confirmation_valid')):
        return {}
    enabled = mode == 'bounded' or feedback_mode == 'integrated'
    plan = plan_repair(review, debate, enabled=enabled)
    if plan and mode == 'bounded':
        plan['origin'] = 'tribunal_only_bounded_repair_v1'
    return plan


def _previous_questions(debate):
    questions = list((debate.get('tribunal_semantic_questions') or {}).get('questions', []))
    for role in ('agent1_critique', 'agent2_critique'):
        response = debate.get(role) or {}
        for item in response.get('question_answers', []):
            questions.extend(str(item.get(k, '')) for k in ('question', 'question_text') if item.get(k))
    return questions


def plan_repair(review, debate=None, enabled=True):
    """Return a single capability-appropriate check, never a generic retry."""
    if not enabled or review.get('_execution_status') == 'FAILED' or not review.get('_format_valid'):
        return {}
    budget = review.get('_case_budget') or {}
    proof = review.get('_independent_verification') or {}
    obligations = proof.get('obligations') or {}
    calls = list(obligations.values()) + list(proof.get('calls') or [])
    def invalid_execution(call):
        if call.get('_execution_status') == 'FAILED':
            return True
        if call.get('_format_valid') is not False:
            return False
        # A complete but self-contradictory audit is a semantic repair target.
        # Never extend this exemption to truncation, invalid JSON or timeouts.
        if call is obligations.get('arguments') and call.get('_contract_error_kind') == 'AUDIT_INCOMPLETE':
            from engine.evidence_review_v5 import complete_audit_response
            return not complete_audit_response(call)
        return True
    if any(invalid_execution(x) for x in calls):
        # Execution/format recovery already has its bounded allowance. It must
        # not silently turn into an extra semantic hearing.
        return {}
    visual, mapping, challenge = (obligations.get(k) or {} for k in ('visual', 'mapping', 'arguments'))
    from engine.tribunal_process import audit_inconsistency
    inconsistent = audit_inconsistency(dict(challenge, _format_valid=True),
        (proof.get('calls') or [{}])[0].get('relation'))
    stop = proof.get('stopped_after', '')
    target, question, issue, diagnostic = '', '', '', ''
    if stop == 'visual_grounding':
        question = visual.get('missing_observation', '')
        if not question:
            unsupported = [x for x in visual.get('observations', []) if x.get('supported') is False]
            if unsupported:
                question = 'What exact visible state and attachment occur for this disputed observation: ' + unsupported[0].get('observed', '')
        target, issue = 'VISUAL_PREMISE', 'missing_or_disputed_observation'
    elif stop == 'entity_scope_mapping' and mapping.get('unmatched_roles'):
        diagnostic = '; '.join(mapping['unmatched_roles'])
        question = 'Which caption participant corresponds to the pictured participant for this disputed role: ' + diagnostic
        target, issue = 'COUNTER_INTERPRETATION', 'role_scope_mapping'
    elif stop == 'relation_direction' and proof.get('calls'):
        decision = proof['calls'][0]
        diagnostic = '; '.join(decision.get('unestablished_conditions', [])) or decision.get('reason', '')
        question = 'What deciding image-to-caption evidence resolves this specific condition or inference: ' + diagnostic
        target, issue = 'COUNTER_INTERPRETATION', 'unresolved_relation'
    elif inconsistent:
        import json
        diagnostic = json.dumps({'inconsistency': inconsistent,
            'original_audit': {k: v for k, v in challenge.items() if not k.startswith('_')}}, ensure_ascii=True)
        question = 'Verify whether a material competing reading exists; reconcile the alternative, its relation and status from the source evidence.'
        target, issue = 'COUNTER_INTERPRETATION', 'contradictory_audit'
    elif challenge.get('role_scope_errors') or challenge.get('decision_errors') or challenge.get('alternative_status') == 'UNRESOLVED':
        diagnostic = '; '.join(challenge.get('role_scope_errors', []) + challenge.get('decision_errors', [])) or challenge.get('alternative', '')
        question = 'Check this disputed image-to-caption inference against the sources and identify the distinguishing fact: ' + diagnostic
        target, issue = 'COUNTER_INTERPRETATION', 'disputed_inference'
        if (challenge.get('role_scope_errors') or challenge.get('decision_errors')) and review.get('_compact_value'):
            issue = 'audit_objection'
    elif any(c.get('status') != 'PASS' for c in challenge.get('process_checks', {}).values()):
        diagnostic = '; '.join(name + ': ' + c.get('reason', '') for name, c in challenge['process_checks'].items()
                               if c.get('status') != 'PASS')
        question = 'Check this disputed source-to-image inference and identify the deciding evidence: ' + diagnostic
        target, issue = 'COUNTER_INTERPRETATION', 'disputed_inference'
    elif proof.get('calls') and proof['calls'][0].get('relation') != review.get('relation'):
        diagnostic = proof['calls'][0].get('reason', '')
        question = 'Compare the caption and actual image state for this disputed inference; identify the deciding fact: ' + diagnostic
        target, issue = 'COUNTER_INTERPRETATION', 'relation_disagreement'
    elif review.get('relation') == 'UNRESOLVED':
        from engine.review_routing import observable_followup
        original_question = review.get('targeted_question', '')
        inspection = observable_followup(original_question)
        routed_review = dict(review)
        if inspection:
            routed_review.update(targeted_question=inspection, requested_follow_up='VISUAL_PREMISE')
            diagnostic = original_question
        routed, routing_audit = route_followup(routed_review)
        if inspection:
            for entry in routing_audit:
                entry['original_question'] = original_question
                entry['operation'] = 'inspect_observable_part_only'
            review['_follow_up_routing_audit'] = routing_audit
        if routed['blocked']:
            review['_follow_up_question_status'] = 'NO_NEW_AVAILABLE_CHECK'
            review['_follow_up_routing_audit'] = routing_audit
        for role in ('visual', 'language', 'tribunal'):
            if routed[role]:
                question = routed[role][0]
                target = {'visual': 'VISUAL_PREMISE', 'language': 'CAPTION_PREMISE', 'tribunal': 'COUNTER_INTERPRETATION'}[role]
                break
        if not question and routed['blocked'] and review.get('_process_audit_version') == 'process-audit-1':
            # Investigate necessity; do not send unavailable history to a witness.
            diagnostic = routed['blocked'][0]
            question = 'Is the unresolved source condition necessary for this image-to-caption relation, or can the relation be established without it?'
            target = 'COUNTER_INTERPRETATION'
        issue = 'unresolved_condition'
    if not question:
        return {}
    # Witness routes cap questions at 320 characters. A semantic repair carries
    # its complete diagnostic separately; never clip it or feed it to a witness.
    if target == 'COUNTER_INTERPRETATION' and diagnostic and len(question) > 320:
        question = ('Check the disputed image-to-caption inference in the attached repair context; '
                    'identify the deciding evidence and whether the criticism is valid.')
    relevance_check = (review.get('_process_audit_version') == 'process-audit-1'
                       and issue in {'unresolved_condition', 'unresolved_relation'})
    reserve = repair_reserve(proof, issue)
    role, reason = route_question(question, target)
    freeze_candidate = (role == 'tribunal' and issue in FROZEN_REPAIR_ISSUES
                        and review.get('_compact_eligible'))
    required = reserve + (0.0 if freeze_candidate else 45.0)
    if budget.get('remaining_seconds') is not None and budget['remaining_seconds'] < required:
        review['_follow_up_budget_status'] = 'SKIPPED_INSUFFICIENT_REMAINING_BUDGET'
        review['_follow_up_budget_requirement_seconds'] = required
        return {}
    if role == 'blocked' or not new_semantic_questions([question], _previous_questions(debate or {})):
        review['_follow_up_question_status'] = 'NO_NEW_AVAILABLE_CHECK'
        return {}
    # One failed obligation -> one route. The diagnostic is fallible, not truth.
    plan = {'status': 'MEDIATE', 'provisional_verdict': 'ABSTAIN', 'confidence': 0.0,
        'agent1_questions': [question] if role == 'visual' else [],
        'agent2_questions': [question] if role == 'language' else [],
        'verification_requests': [question] if role == 'tribunal' else [],
        '_format_valid': True, '_usable': True, '_invalid_evidence_ids': [],
        '_valid_evidence_ids': list(review.get('_valid_evidence_ids') or []),
        'repair_reasons': [issue], 'origin': 'integrated_diagnostic_repair_v1',
        'repair_context': {'failed_requirement': issue, 'disputed_detail': diagnostic,
            'question': question, 'route': role,
                'instruction': 'Perform this specific check using the sources and any new testimony. The criticism may be wrong. Preserve valid observations and source meaning. Do not repeat an unavailable request or force a binary answer. If revising an objection, explain in the audit reason which original objection was mistaken and why; never erase an objection merely to obtain acceptance.'
                    + (' Test whether plausible alternatives for the missing condition could change the relation while known evidence stays fixed. '
                       'If yes or unclear, retain uncertainty. Unavailable information can be essential; hypothetical alternatives are not observations.' if relevance_check else ''),
                'uncertainty_relevance_check': relevance_check},
        'routing_audit': {'target': target, 'role': role, 'reason': reason}}
    if freeze_candidate:
        from copy import deepcopy
        plan['repair_context']['frozen_candidate'] = {
            'value': deepcopy(review['_compact_value']),
            'source_sha256': (review.get('_judge_packet') or {}).get('source_sha256', ''),
            'image_sha256': review.get('_case_image_sha256', '')}
    return plan


def repair_reserve(proof, issue):
    """Reserve the whole remaining proof using this case's observed speed.

    The 30-second floor is the existing per-stage allocation. Slower completed
    calls raise it for this case only; sample order and other cases never do.
    A failed/contradictory result cannot be counted as a reusable obligation.
    """
    obligations = proof.get('obligations') or {}
    reusable = 0
    if issue in {'role_scope_mapping', 'unresolved_relation', 'disputed_inference', 'relation_disagreement', 'contradictory_audit', 'audit_objection'}:
        visual = obligations.get('visual') or {}
        if visual.get('verified') and visual.get('_cache_key'):
            reusable += 1
            mapping = obligations.get('mapping') or {}
            if issue != 'role_scope_mapping' and mapping.get('verified') and mapping.get('_cache_key'):
                reusable += 1
    calls = list(obligations.values()) + list(proof.get('calls') or [])
    durations = []
    for call in calls:
        if call.get('_execution_status') != 'SUCCEEDED' or call.get('_cache_hit'):
            continue
        diagnostics = call.get('_generation_diagnostics') or []
        if isinstance(diagnostics, dict):
            diagnostics = [diagnostics]
        attempts = [d.get('elapsed_seconds', 0) for d in diagnostics if d.get('elapsed_seconds', 0) > 0]
        seconds = max(attempts, default=call.get('_generation_seconds', 0))
        if seconds > 0:
            durations.append(float(seconds))
    per_stage = max([30.0] + durations)
    return (4 - reusable) * per_stage


def restore_proof_cache(runtime, review, ledger, language, caption, context):
    """Reuse source-bound obligations across model unload/reload or checkpoint resume."""
    from copy import deepcopy
    from engine.semantic_bridge import build_semantic_bridge
    from engine.evidence_review_v5 import audit
    from engine.evidence_verification import executed
    proposal = build_semantic_bridge(review, ledger, dict((language or {}).get('claim_contract', {}), source_caption=caption), language)
    proof = proposal.get('independent_verification') or {}
    if not audit(proposal, ledger)['bound_to_current_case']:
        return 120.0
    # Keys bind model/profile, exact prompt/schema, complete catalogue and image.
    # No old result can match a changed downstream request.
    cache = getattr(runtime, '_evidence_call_cache', {})
    for call in list((proof.get('obligations') or {}).values()) + list(proof.get('calls') or []):
        if executed(call) and call.get('_semantic_contract_valid') is not False and call.get('_cache_key'):
            if len(cache) >= 128:
                cache.pop(next(iter(cache)))
            cache[call['_cache_key']] = deepcopy(call)
    runtime._evidence_call_cache = cache
    return repair_reserve(proof, context.get('failed_requirement'))
