"""Separate execution, syntax and semantic outcomes without consulting gold."""


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
    retrieval = (review.get("_retrieval_audit") or {}).get("prior_review")
    if retrieval:
        prior.append(retrieval)
    children = [review_timing(item) for item in prior]
    total = review.get("_generation_seconds")
    own_verification = (review.get("_independent_verification") or {}).get("_generation_seconds", 0.0)
    verification = own_verification + sum(c["verification_seconds"] for c in children)
    diagnostics = review.get("_generation_diagnostics") or []
    if isinstance(diagnostics, dict):
        diagnostics = [diagnostics]
    timeout = sum(float(d.get("elapsed_seconds", 0.0)) for d in diagnostics
                  if d.get("termination_reason") == "TIMEOUT" or d.get("timeout_seconds") is not None)
    proof = review.get("_independent_verification") or {}
    proof_calls = list(proof.get("calls") or []) + list((proof.get("obligations") or {}).values())
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


def classify_review(review):
    review = review or {}
    error = str(review.get("_format_error") or "")
    execution = review.get("_execution_status")
    if execution is None:
        execution = ("FAILED" if "generation_failed:" in error else
                     "SUCCEEDED" if review else "NOT_RUN")
    valid = bool(review.get("_format_valid", False))
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
        verification_status = ("BLOCKED_IMAGE_BINDING" if verification.get("input_binding_error") else
            "EXECUTED" if len(calls) == 2 and all(c.get("_execution_status") == "SUCCEEDED"
                and c.get("_format_valid") for c in calls + required) else "INCOMPLETE_VERIFICATION")
    else:
        verification_status = ("NOT_RUN" if not review else "BLOCKED_EXECUTION" if execution == "FAILED" else
            "BLOCKED_CONTEXT" if review.get("_context_valid") is False else
            "BLOCKED_OUTPUT" if not valid else "UNRESOLVED_PROPOSAL" if review.get("relation") == "UNRESOLVED" else
            "NOT_REQUESTED")
    return {
        "execution_status": execution,
        "execution_error_type": review.get("_execution_error_type") or (
            execution_error_type(error) if execution == "FAILED" else None),
        "schema_status": "VALID" if valid else "TRUNCATED" if review.get("_output_status") == "TRUNCATED" else "INVALID" if execution == "SUCCEEDED" else "NOT_PRODUCED",
        "context_status": review.get("_context_status", "NOT_CHECKED"),
        "verification_status": verification_status,
        "semantic_eligible": eligible,
        "semantic_judgment": verdict,
        "semantic_judgment_valid": verdict in {"ENTAILS", "CONTRADICTS"},
        "semantic_abstained": verdict == "ABSTAIN",
        "terminal_output_policy": "ELIGIBLE_FOR_EVIDENCE_GATE" if verdict in {"ENTAILS", "CONTRADICTS"}
            else "PRESERVE_INITIAL_WITH_EXPLICIT_REVIEW_FAILURE" if not eligible
            else "PRESERVE_INITIAL_WITH_SEMANTIC_UNCERTAINTY",
    }
