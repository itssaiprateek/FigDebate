"""Routing and deterministic acceptance gate for the optional Qwen judge."""

from engine.evidence_ledger import attach_evidence_audit
from engine.review_board import (
    attach_final_review,
    decision_grade_strength,
    review_revision,
)


JUDGE_MODES = {"disabled", "shadow", "appellate", "mediated", "tribunal"}
JUDGE_SCOPES = {"escalated", "all"}
MIN_APPELLATE_CONFIDENCE = 0.75
RELATION_FOR_LABEL = {"ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"}


def judge_request_reasons(result, scope="escalated"):
    if scope not in JUDGE_SCOPES:
        raise ValueError(f"Unknown judge scope: {scope}")
    if scope == "all":
        return ["all_samples_scope"]

    reasons = []
    decision = result.get("decision", {}) or {}
    review = decision.get("_review_board", {}) or {}
    comparison = result.get("comparison", {}) or {}
    if result.get("debate_triggered", False):
        reasons.append("debate_completed")
    if not decision.get("_final_decision_valid", False):
        reasons.append("invalid_primary_decision")
    if not review.get("directionally_grounded", False):
        reasons.append("final_decision_not_directionally_grounded")
    confidence = decision.get("confidence")
    if isinstance(confidence, (int, float)) and confidence <= 0.60:
        reasons.append("low_final_confidence")
    if comparison.get("direct_support_count", 0) and comparison.get(
        "direct_conflict_count", 0
    ):
        reasons.append("mixed_directional_evidence")
    return list(dict.fromkeys(reasons))


def judge_feedback_candidate(judgment, previous_label):
    """Route disagreement to verification without creating pseudo-label memory."""
    recorded = bool(
        judgment.get("_format_valid", False)
        and judgment.get("verdict") in {"ENTAILS", "CONTRADICTS"}
        and judgment.get("verdict") != previous_label
    )
    return {
        "recorded": recorded,
        "role": "human_or_gold_verified_review_only",
        "memory_update_applied": False,
        "reason": (
            "independent_judge_disagreement"
            if recorded else "judge_agreement_or_abstention"
        ),
    }


def _cited_texts(ledger, cited_ids):
    cited = set(cited_ids or [])
    return [
        str(item.get("text", "")).strip()
        for item in (ledger or [])
        if item.get("id") in cited and str(item.get("text", "")).strip()
    ]


def _decision_grade_ids(ledger, relation):
    return {
        item.get("id")
        for item in (ledger or [])
        if item.get("grounded", False)
        and item.get("relation") == relation
        and (
            item.get("decision_grade", False)
            or item.get("verification", {}).get("decision_grade", False)
        )
    }


def apply_judge_review(
    current_decision,
    judgment,
    ledger,
    claim_contract=None,
    mode="shadow",
):
    """Apply no change unless the existing review board accepts the proposal."""
    if mode not in JUDGE_MODES - {"disabled"}:
        raise ValueError(f"Judge review requires shadow or appellate mode, got: {mode}")
    metadata = {
        "mode": mode,
        "accepted": False,
        "changed_decision": False,
        "previous_label": current_decision.get("label"),
        "previous_confidence": current_decision.get("confidence"),
        "proposed_label": judgment.get("verdict"),
        "reason": "",
        "candidate_evidence_audit": {},
    }
    if not judgment.get("_format_valid", False):
        metadata["reason"] = "invalid_judge_contract"
        return current_decision, metadata
    if mode == "shadow":
        metadata["reason"] = "shadow_mode_never_changes_decisions"
        return current_decision, metadata
    if judgment.get("verdict") == "ABSTAIN":
        metadata["reason"] = "judge_abstained"
        return current_decision, metadata
    if judgment.get("verdict") == current_decision.get("label"):
        metadata["reason"] = "judge_agrees_with_current_decision"
        return current_decision, metadata
    if judgment.get("confidence", 0.0) < MIN_APPELLATE_CONFIDENCE:
        metadata["reason"] = "judge_confidence_below_appellate_threshold"
        return current_decision, metadata
    if judgment.get("_invalid_evidence_ids"):
        metadata["reason"] = "judge_cited_unknown_evidence"
        return current_decision, metadata
    cited_ids = judgment.get("_valid_evidence_ids", [])
    if not cited_ids:
        metadata["reason"] = "judge_cited_no_current_image_evidence"
        return current_decision, metadata
    if not judgment.get("visual_observations"):
        metadata["reason"] = "judge_reported_no_direct_visual_observation"
        return current_decision, metadata

    proposed_label = judgment["verdict"]
    proposed_grade_ids = _decision_grade_ids(
        ledger, RELATION_FOR_LABEL[proposed_label]
    ) & set(cited_ids)
    current_grade_ids = _decision_grade_ids(
        ledger, RELATION_FOR_LABEL.get(current_decision.get("label"))
    )
    if not proposed_grade_ids:
        metadata["reason"] = "judge_citations_lack_decision_grade_direction"
        return current_decision, metadata
    proposed_strength = decision_grade_strength(
        ledger,
        RELATION_FOR_LABEL[proposed_label],
        evidence_ids=proposed_grade_ids,
    )
    current_strength = decision_grade_strength(
        ledger, RELATION_FOR_LABEL.get(current_decision.get("label"))
    )
    if current_grade_ids and proposed_strength <= current_strength:
        metadata["reason"] = "unresolved_opposing_decision_grade_evidence"
        return current_decision, metadata
    metadata["proposed_evidence_strength"] = proposed_strength
    metadata["current_evidence_strength"] = current_strength

    cited_texts = _cited_texts(ledger, cited_ids)
    candidate = dict(current_decision)
    candidate.update({
        "label": proposed_label,
        "confidence": min(float(judgment["confidence"]), 0.85),
        "explanation": judgment.get("reason", ""),
        "decision_method": "multimodal_judge_appellate",
        "debate_needed": False,
        "_model_cited_evidence_ids": list(cited_ids),
        "_final_decision_valid": True,
    })
    if proposed_label == "ENTAILS":
        candidate["visual_support"] = cited_texts
    else:
        candidate["contradictions"] = cited_texts

    accepted, reason, candidate_audit = review_revision(
        current_decision,
        candidate,
        ledger,
        visual_review=None,
        claim_contract=claim_contract,
    )
    metadata["reason"] = reason
    metadata["candidate_evidence_audit"] = candidate_audit
    if not accepted:
        return current_decision, metadata

    candidate = attach_evidence_audit(candidate, ledger)
    candidate = attach_final_review(candidate, ledger, claim_contract)
    metadata["accepted"] = True
    metadata["changed_decision"] = True
    return candidate, metadata
