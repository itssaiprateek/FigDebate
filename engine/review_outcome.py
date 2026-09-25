"""Separate execution, syntax and semantic outcomes without consulting gold."""
from enum import Enum


class ExecutionStatus(str, Enum):
    NOT_RUN = 'NOT_RUN'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'


class ValidationStatus(str, Enum):
    NOT_PRODUCED = 'NOT_PRODUCED'
    VALID = 'VALID'
    INVALID = 'INVALID'
    TRUNCATED = 'TRUNCATED'


class SemanticStatus(str, Enum):
    NOT_ASSESSED = 'NOT_ASSESSED'
    UNRESOLVED = 'UNRESOLVED'
    CONTRADICTORY = 'CONTRADICTORY'
    DISAGREEMENT = 'DISAGREEMENT'
    CHECK_NOT_PASSED = 'CHECK_NOT_PASSED'
    ASSESSED = 'ASSESSED'


def outcome_dimensions(call, status):
    """Orthogonal observations; ASSESSED never asserts objective correctness."""
    execution = (ExecutionStatus.NOT_RUN if not call else ExecutionStatus.FAILED
                 if call.get('_execution_status') == 'FAILED' else ExecutionStatus.COMPLETED)
    validation = (ValidationStatus.NOT_PRODUCED if execution != ExecutionStatus.COMPLETED else
                  ValidationStatus.TRUNCATED if call.get('_output_status') == 'TRUNCATED' else
                  ValidationStatus.VALID if call.get('_format_valid') is True
                  or call.get('_contract_error_kind') in {'SEMANTIC_INCONSISTENCY', 'AUDIT_INCOMPLETE'} else ValidationStatus.INVALID)
    if execution == ExecutionStatus.COMPLETED and call.get('response_valid') is False:
        validation = ValidationStatus.INVALID
    semantic = SemanticStatus.NOT_ASSESSED
    if execution == ExecutionStatus.COMPLETED and validation == ValidationStatus.VALID:
        semantic = {
            'CONTRADICTORY_AUDIT': SemanticStatus.CONTRADICTORY,
            'INCONSISTENT_JUDGMENT': SemanticStatus.CONTRADICTORY,
            'RELATION_DISAGREEMENT': SemanticStatus.DISAGREEMENT,
            'UNRESOLVED_EVIDENCE': SemanticStatus.UNRESOLVED,
            'UNRESOLVED_ARGUMENT': SemanticStatus.UNRESOLVED,
            'SEMANTIC_CHECK_NOT_PASSED': SemanticStatus.CHECK_NOT_PASSED,
            'COMPLETED': SemanticStatus.ASSESSED,
        }.get(status, SemanticStatus.NOT_ASSESSED)
    return {'execution': execution.value, 'validation': validation.value, 'semantic': semantic.value}


def resolution_category(review, gate):
    """Describe the observed terminal path, not ground-truth correctness."""
    if gate.get('accepted'):
        return 'SUPPORTED_CORRECTION'
    if gate.get('confirmation_valid'):
        return 'SUPPORTED_AGREEMENT'
    stages = stage_outcomes(review)['stage_outcomes']
    if any(s['status'] in {'RUNTIME_FAILURE', 'INVALID_OUTPUT', 'CONTEXT_BLOCKED'} for s in stages):
        return 'EXECUTION_OR_CONTRACT_FAILURE'
    if any(s['status'] == 'INCOMPLETE_AUDIT' for s in stages):
        return 'INCOMPLETE_AUDIT'
    proof = review.get('_independent_verification') or {}
    if proof.get('input_binding_error'):
        return 'EXECUTION_OR_CONTRACT_FAILURE'
    obligations = proof.get('obligations') or {}
    visual = obligations.get('visual') or {}
    relation = (proof.get('calls') or [{}])[0]
    if (visual.get('coverage_complete') is False or relation.get('unestablished_conditions')
            or any(c.get('unestablished_condition') for c in review.get('claim_checks', []))):
        return 'MISSING_EVIDENCE'
    if (review.get('relation') == 'UNRESOLVED' or review.get('provisional_verdict') == 'ABSTAIN'
            or any(s['status'] in {'UNRESOLVED_ARGUMENT', 'RELATION_DISAGREEMENT',
                                  'CONTRADICTORY_AUDIT', 'UNRESOLVED_EVIDENCE', 'INCONSISTENT_JUDGMENT'} for s in stages)):
        return 'UNRESOLVED_INTERPRETATION'
    return 'UNVERIFIED_PROPOSAL'


def decision_path(review):
    """Persist interpretation and source references separately from observations."""
    from copy import deepcopy
    proof = review.get('_independent_verification') or {}
    obligations = proof.get('obligations') or {}
    relation = (proof.get('calls') or [{}])[0]
    return {
        'source_identity_checked': (review.get('_communication_audit') or {}).get('source_caption_unchanged'),
        'delivery': deepcopy((review.get('_prompt_budget') or {}).get('context_delivery')),
        'selected_observation_ids': list(proof.get('selected_evidence_ids') or []),
        'caption_bindings': deepcopy((obligations.get('mapping') or {}).get('bindings', [])),
        'proposed_inference': review.get('semantic_bridge', review.get('reason', '')),
        'independent_condition_checks': deepcopy(relation.get('condition_checks', [])),
        'challenge': {key: deepcopy((obligations.get('arguments') or {}).get(key)) for key in
                      ('alternative', 'alternative_status', 'decision_errors', 'role_scope_errors', 'reason')},
        'stage_inputs': [{'stage': c.get('obligation'), 'input_sha256': c.get('input_sha256'),
                          'repair': deepcopy(c.get('_field_repair_audit'))}
                         for c in list(obligations.values()) + list(proof.get('calls') or [])
                         if c.get('input_sha256')],
        'execution': stage_outcomes(review),
        'semantic_truth_established': False,
    }


def hearing_accounting(judge):
    """Count observed transitions, never infer execution from a planned repair."""
    session = judge.get("tribunal_session") or {}
    transitions = session.get("hearing_transitions") or []
    plan = judge.get("verification_followup_plan") or {}
    repairs = [t for t in transitions if t.get("repair_reasons", plan.get("repair_reasons", []))]
    return {
        "tribunal_repair_planned": bool(plan.get("repair_reasons")),
        "tribunal_followup_hearing_count": len(transitions),
        "tribunal_repair_hearing_count": len(repairs),
        "tribunal_content_changed_count": sum(t.get("before") is not None and t.get("after") is not None
            and t["before"] != t["after"] for t in transitions),
        # Count actual reviews, not the transition's scheduling intention.
        "tribunal_rereview_count": max(0, len(judge.get("tribunal_reviews") or []) - 1),
        "tribunal_repair_followup_attempted": bool(repairs),
        "tribunal_repair_reasons": " | ".join(sorted({r for t in repairs
            for r in t.get("repair_reasons", plan.get("repair_reasons", []))})),
    }


def execution_error_type(error):
    text = str(error).casefold()
    if 'host suspended' in text:
        return 'HOST_INTERRUPTED'
    if "case budget" in text:
        return "CASE_BUDGET"
    if "out of memory" in text or "vram profile" in text:
        return "OUT_OF_MEMORY"
    if "context budget" in text or "token budget" in text:
        return "CONTEXT_BUDGET"
    if isinstance(error, TimeoutError) or "timed out" in text:
        return "TIMEOUT"
    return "RUNTIME_ERROR"


def review_timing(review):
    """Disjoint phase totals; timeout elapsed is an overlapping diagnostic.

    Existing review seconds include verification and explicitly archived prior
    attempts. Never recursively walk arbitrary duplicated trace dictionaries.
    """
    review = review or {}
    prior = list(review.get("_verification_repair_history") or [])
    prior.extend(review.get("_feedback_review_history") or [])
    retrieval = (review.get("_retrieval_audit") or {}).get("prior_review")
    if retrieval:
        prior.append(retrieval)
    children = [review_timing(item) for item in prior]
    total = review.get("_generation_seconds")
    own_verification = (review.get("_independent_verification") or {}).get("_generation_seconds", 0.0)
    own_verification += (review.get("_argument_repair") or {}).get("_generation_seconds", 0.0)
    verification = own_verification + sum(c["verification_seconds"] for c in children)
    diagnostics = review.get("_generation_diagnostics") or []
    if isinstance(diagnostics, dict):
        diagnostics = [diagnostics]
    timeout = sum(float(d.get("elapsed_seconds", 0.0)) for d in diagnostics
                  if d.get("termination_reason") == "TIMEOUT" or d.get("timeout_seconds") is not None)
    proof = review.get("_independent_verification") or {}
    proof_calls = list(proof.get("calls") or []) + list((proof.get("obligations") or {}).values())
    if review.get("_argument_repair"):
        proof_calls.append(review["_argument_repair"])
    timeout += sum(review_timing(call)["timeout_elapsed_seconds"] for call in proof_calls)
    timeout += sum(c["timeout_elapsed_seconds"] for c in children)
    return {"inclusive_seconds": total, "verification_seconds": verification,
            "proposal_and_format_retry_seconds": total - verification if total is not None and total >= verification else None,
            "timeout_elapsed_seconds": timeout,
            "timeout_is_subset_not_additive": True,
            "phase_attribution_complete": total is not None and total >= verification}


def failed_review(error, stage, elapsed=0.0):
    return {"_format_valid": False, "_execution_status": "FAILED",
            "_execution_error_type": execution_error_type(error),
            "_format_error": "generation_failed:" + str(error),
            "_generation_seconds": max(float(elapsed), 0.000001),
            "_generation_diagnostics": {"stage": stage, "error": str(error)},
            "_valid_evidence_ids": [], "_invalid_evidence_ids": []}


def stage_outcomes(review):
    """Report only observed work; downstream omissions are not extra failures."""
    proof = review.get('_independent_verification') or {}
    obligations = proof.get('obligations') or {}
    relation = (proof.get('calls') or [{}])[0]
    stages = [('proposal', review), ('visual', obligations.get('visual', {})),
              ('mapping', obligations.get('mapping', {})), ('relation', relation),
              ('arguments', obligations.get('arguments', {}))]
    rows = []
    stop_alias = {'visual_grounding': 'visual', 'entity_scope_mapping': 'mapping',
                  'relation_direction': 'relation'}
    stopped = stop_alias.get(proof.get('stopped_after'), proof.get('stopped_after'))
    from engine.tribunal_process import audit_inconsistency
    for name, call in stages:
        error = call.get('_execution_error_type')
        if not call:
            status = 'NOT_RUN'
        elif call.get('_output_status') == 'TRUNCATED':
            status, error = 'INVALID_OUTPUT', 'OUTPUT_TRUNCATED'
        elif call.get('_execution_status') == 'FAILED':
            status = 'RUNTIME_FAILURE'
        elif call.get('_contract_error_kind') == 'SEMANTIC_INCONSISTENCY':
            status = 'INCONSISTENT_JUDGMENT'
        elif call.get('_contract_error_kind') == 'AUDIT_INCOMPLETE':
            status = 'INCOMPLETE_AUDIT'
        elif call.get('_format_valid') is not True:
            status = 'INVALID_OUTPUT'
        elif name == 'mapping' and call.get('response_valid') is False:
            status = 'INVALID_OUTPUT'
        elif name == 'proposal' and call.get('_context_valid') is False:
            status = 'CONTEXT_BLOCKED'
        elif call.get('relation') == 'UNRESOLVED' or call.get('provisional_verdict') == 'ABSTAIN':
            status = 'UNRESOLVED_EVIDENCE'
        elif name == 'mapping' and call.get('unmatched_roles'):
            status = 'UNRESOLVED_EVIDENCE'
        elif name == 'relation' and review.get('relation') in {'SUPPORT', 'CONFLICT'} and call.get('relation') != review['relation']:
            status = 'RELATION_DISAGREEMENT'
        elif name == 'arguments' and audit_inconsistency(call, relation.get('relation')):
            status = 'CONTRADICTORY_AUDIT'
        elif name == 'arguments' and (call.get('decision_errors') or call.get('role_scope_errors')
                                     or call.get('alternative_status') == 'UNRESOLVED'
                                     or any(c.get('status') != 'PASS' for c in call.get('process_checks', {}).values())):
            status = 'UNRESOLVED_ARGUMENT'
        elif name == stopped:
            status = 'SEMANTIC_CHECK_NOT_PASSED'
        else:
            status = 'COMPLETED'  # Completion is not proof of semantic correctness.
        rows.append({'stage': name, 'status': status, 'error_type': error,
                     'detail': call.get('_semantic_error') or call.get('_format_error', ''),
                     **outcome_dimensions(call, status)})
    first = next((r['stage'] for r in rows if r['status'] not in {'COMPLETED', 'NOT_RUN'}), None)
    if not first and proof.get('input_binding_error'):
        first = 'case_binding'
    return {'stage_outcomes': rows, 'first_blocking_stage': first}


def classify_review(review):
    review = review or {}
    error = str(review.get("_format_error") or "")
    execution = review.get("_execution_status")
    if execution is None:
        execution = ("FAILED" if "generation_failed:" in error else
                     "SUCCEEDED" if review else "NOT_RUN")
    valid = bool(review.get("_format_valid", False))
    contradictory = review.get('_contract_error_kind') == 'SEMANTIC_INCONSISTENCY'
    eligible = execution == "SUCCEEDED" and valid and review.get("_context_valid", True)
    verdict = review.get("best_semantic_judgment", review.get(
        "provisional_verdict", review.get("verdict")))
    verdict = verdict if eligible and verdict in {"ENTAILS", "CONTRADICTS", "ABSTAIN"} else None
    verification = review.get("_independent_verification")
    if verification is not None:
        calls = verification.get("calls", [])
        obligations = verification.get("obligations") or {}
        required = [obligations.get(name, {}) for name in ("visual", "mapping", "arguments")]
        caption = obligations.get("caption") or {}
        if caption.get("method") != "exact_source_identity":
            required.append(caption)
        if verification.get("schema_version") in {"4.0", "5.0"}:
            calls = verification.get("calls", [])
            required = [obligations.get(name, {}) for name in ("visual", "mapping", "arguments")]
            expected_calls = 1
        else:
            expected_calls = 2
        verification_status = ("BLOCKED_IMAGE_BINDING" if verification.get("input_binding_error") else
            "EXECUTED" if len(calls) == expected_calls and all(c.get("_execution_status") == "SUCCEEDED"
                and c.get("_format_valid") for c in calls + required) else "INCOMPLETE_VERIFICATION")
        actual = [c for c in calls + required if '_execution_status' in c]
        if any(c.get('_execution_status') == 'FAILED' for c in actual):
            verification_status = 'FAILED_VERIFICATION_EXECUTION'
        elif any((c.get('_format_valid') is False or c.get('response_valid') is False)
                 and c.get('_contract_error_kind') not in {'SEMANTIC_INCONSISTENCY', 'AUDIT_INCOMPLETE'} for c in actual):
            verification_status = 'INVALID_VERIFICATION_OUTPUT'
        elif any(c.get('_contract_error_kind') == 'AUDIT_INCOMPLETE' for c in actual):
            verification_status = 'INCOMPLETE_VERIFICATION_AUDIT'
        elif any(c.get('_contract_error_kind') == 'SEMANTIC_INCONSISTENCY' for c in actual):
            verification_status = 'INCONSISTENT_VERIFICATION_JUDGMENT'
        elif verification.get('stopped_after') and verification_status == 'INCOMPLETE_VERIFICATION':
            verification_status = 'SEMANTICALLY_REJECTED'
    else:
        verification_status = ("NOT_RUN" if not review else "BLOCKED_EXECUTION" if execution == "FAILED" else
            "BLOCKED_CONTEXT" if review.get("_context_valid") is False else
            "BLOCKED_OUTPUT" if not valid else "UNRESOLVED_PROPOSAL" if review.get("relation") == "UNRESOLVED" else
            "NOT_REQUESTED")
    verification_failed = verification_status in {'FAILED_VERIFICATION_EXECUTION', 'INVALID_VERIFICATION_OUTPUT', 'INCOMPLETE_VERIFICATION', 'BLOCKED_IMAGE_BINDING'}
    contradictory = contradictory or verification_status == 'INCONSISTENT_VERIFICATION_JUDGMENT'
    incomplete_audit = verification_status == 'INCOMPLETE_VERIFICATION_AUDIT'
    terminal = ('INCOMPLETE_AUDIT' if incomplete_audit else
                'EXECUTION_INTERRUPTED_OR_FAILED' if execution == 'FAILED' or verification_failed else
                'CONTRADICTORY_JUDGMENT' if contradictory else
                'EXECUTION_INTERRUPTED_OR_FAILED' if not eligible else
                'SEMANTIC_UNCERTAINTY' if verdict == 'ABSTAIN' or verification_status == 'INCONSISTENT_VERIFICATION_JUDGMENT'
                else 'PROPOSAL_REQUIRES_GATE')
    return {
        **stage_outcomes(review),
        'terminal_outcome': terminal,
        "execution_status": execution,
        "execution_error_type": review.get("_execution_error_type") or (
            execution_error_type(error) if execution == "FAILED" else None),
        "schema_status": "VALID" if valid or (execution == 'SUCCEEDED' and review.get('_contract_error_kind') == 'SEMANTIC_INCONSISTENCY') else "TRUNCATED" if review.get("_output_status") == "TRUNCATED" else "INVALID" if execution == "SUCCEEDED" else "NOT_PRODUCED",
        "context_status": review.get("_context_status", "NOT_CHECKED"),
        "verification_status": verification_status,
        "semantic_eligible": eligible,
        "semantic_judgment": verdict,
        "semantic_judgment_valid": verdict in {"ENTAILS", "CONTRADICTS"},
        "semantic_abstained": verdict == "ABSTAIN",
        "terminal_output_policy": "PRESERVE_INITIAL_WITH_INCOMPLETE_AUDIT" if incomplete_audit
            else "PRESERVE_INITIAL_WITH_EXPLICIT_REVIEW_FAILURE" if verification_failed
            else "PRESERVE_INITIAL_WITH_INCONSISTENT_REVIEW" if contradictory
            else "ELIGIBLE_FOR_EVIDENCE_GATE" if verdict in {"ENTAILS", "CONTRADICTS"}
            else "PRESERVE_INITIAL_WITH_EXPLICIT_REVIEW_FAILURE" if not eligible
            else "PRESERVE_INITIAL_WITH_SEMANTIC_UNCERTAINTY",
    }
