"""Label-blind deterministic dossier and tribunal escalation policy."""

from engine.question_router import build_question_plan


def build_pre_hearing_audit(decision, comparison, debate_assessment=None):
    """Decide whether a live hearing can add missing evidence.

    The audit never receives a gold label or dataset phenomenon. It records
    concrete contract/evidence deficiencies and closes cases that already
    have an auditable direction unless a real conflict remains.
    """
    decision = decision or {}
    comparison = comparison or {}
    assessment = debate_assessment or {}
    contract = comparison.get("claim_contract", {}) or {}
    reasons = []
    if not contract.get(
        "safe_for_tribunal_reasoning",
        contract.get("safe_for_directional_reasoning", False),
    ):
        reasons.append("claim_contract_requires_repair")
    if comparison.get("relation_binding_required", False) and not comparison.get(
        "relation_binding_observed", False
    ):
        reasons.append("structural_binding_unresolved")
    if comparison.get("required_evidence_status") in {
        "MIXED_VERIFIED_EVIDENCE", "MIXED_RELATION_CANDIDATES",
        "INSUFFICIENT_VISUAL_EVIDENCE",
    }:
        reasons.append("directional_evidence_unresolved")
    review = decision.get("_review_board", {}) or {}
    if review and not review.get("directionally_grounded", False):
        reasons.append("decision_not_directionally_grounded")
    confidence = decision.get("confidence")
    if isinstance(confidence, (int, float)) and confidence <= 0.65:
        reasons.append("low_evidence_derived_confidence")
    if assessment.get("trigger", False):
        reasons.extend(assessment.get("signals", []) or [])
    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    plan = build_question_plan(comparison).to_dict()
    requires_live_hearing = bool(reasons) and bool(plan.get("agent1_question"))
    return {
        "schema_version": "1.0",
        "stage": "PRE_HEARING_AUDIT",
        "requires_live_hearing": requires_live_hearing,
        "reasons": reasons,
        "issue_type": plan.get("issue_type"),
        "question_plan": plan,
        "claim_contract_safe": bool(
            contract.get(
                "safe_for_tribunal_reasoning",
                contract.get("safe_for_directional_reasoning", False),
            )
        ),
        "structural_observation_summary": dict(
            comparison.get("structured_observation_summary", {}) or {}
        ),
        "current_label": decision.get("label"),
        "current_confidence": decision.get("confidence"),
    }
