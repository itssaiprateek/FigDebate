"""Deterministic final review for all FigDebate revision proposals."""

import re

from engine.evidence_ledger import audit_decision


VALID_LABELS = {"ENTAILS", "CONTRADICTS"}
RELATION_FOR_LABEL = {"ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"}


def _decision_grade(item):
    return bool(
        item.get("decision_grade", False)
        or item.get("verification", {}).get("decision_grade", False)
    )


def _grade_ids(ledger, relation):
    return {
        item.get("id") for item in (ledger or [])
        if item.get("grounded", False)
        and item.get("relation") == relation
        and _decision_grade(item)
    }


def audit_final_decision(decision, ledger, claim_contract=None):
    evidence = audit_decision(decision, ledger)
    contract = claim_contract or {}
    if decision.get("label") not in VALID_LABELS:
        status = "INVALID_BINARY_DECISION"
    elif evidence.get("valid"):
        status = "DIRECTIONALLY_GROUNDED"
    elif contract and not contract.get("safe_for_directional_reasoning", False):
        status = "CLAIM_CONTRACT_REVIEW_REQUIRED"
    else:
        status = "BINARY_DECISION_WITHOUT_DIRECTIONAL_PROOF"
    return {
        "status": status,
        "binary_valid": decision.get("label") in VALID_LABELS,
        "directionally_grounded": bool(evidence.get("valid")),
        "source_grounded": bool(evidence.get("source_valid")),
        "evidence_audit": evidence,
        "claim_contract_valid": (
            contract.get("safe_for_directional_reasoning")
            if contract else None
        ),
        "claim_contract_warnings": list(contract.get("warnings", []) or []),
    }


def attach_final_review(decision, ledger, claim_contract=None):
    output = dict(decision or {})
    review = audit_final_decision(
        output, ledger, claim_contract
    )
    confidence = output.get("confidence")
    cap = None
    if isinstance(confidence, (int, float)):
        if not review["directionally_grounded"]:
            cap = 0.55 if review["source_grounded"] else 0.35
        if review.get("claim_contract_valid") is False:
            cap = min(cap if cap is not None else 1.0, 0.55)
        if cap is not None and confidence > cap:
            output["confidence"] = cap
    review.update({
        "confidence_before_review": confidence,
        "confidence_after_review": output.get("confidence"),
        "confidence_cap": cap,
        "confidence_cap_applied": bool(
            cap is not None
            and isinstance(confidence, (int, float))
            and confidence > cap
        ),
    })
    output["_review_board"] = review
    return output


def review_revision(
    original,
    candidate,
    ledger,
    visual_review=None,
    claim_contract=None,
):
    """Accept a change only when stronger current-image evidence supports it."""
    original_label = original.get("label")
    candidate_label = candidate.get("label")
    if candidate_label not in VALID_LABELS:
        return False, "invalid_proposed_label", {}
    same_label = candidate_label == original_label

    review = visual_review or {}
    recommendation = str(review.get("recommendation", "")).upper()
    if recommendation and recommendation not in {candidate_label, "ABSTAIN"}:
        return False, "visual_reviewer_recommends_other_label", {}
    if recommendation == "ABSTAIN":
        return (
            False,
            "unchanged" if same_label else "visual_reviewer_abstained",
            {},
        )
    if review and not review.get("specific_evidence", False):
        return False, "visual_review_lacks_specific_evidence", {}
    reason_text = str(review.get("reason", "")).casefold()
    if any(phrase in reason_text for phrase in (
        "no evidence", "no direct visual evidence", "no clear indication",
        "not shown", "not visible", "cannot be determined",
        "does not directly support", "does not directly contradict",
    )):
        return False, "absence_is_not_decision_evidence", {}

    auditable_candidate = dict(candidate)
    if not auditable_candidate.get("_model_cited_evidence_ids"):
        auditable_candidate["_model_cited_evidence_ids"] = list(
            candidate.get("_evidence_audit", {}).get(
                "source_cited_evidence_ids", []
            )
        )
    candidate_audit = audit_decision(auditable_candidate, ledger)
    if not candidate_audit.get("source_valid"):
        return (
            False,
            "unchanged" if same_label
            else "revision_did_not_cite_current_image_evidence",
            candidate_audit,
        )
    if not candidate_audit.get("valid"):
        return (
            False,
            "unchanged" if same_label
            else "revision_lacks_decision_grade_direction",
            candidate_audit,
        )

    candidate_relation = RELATION_FOR_LABEL[candidate_label]
    original_relation = RELATION_FOR_LABEL.get(original_label)
    candidate_ids = _grade_ids(ledger, candidate_relation)
    original_ids = _grade_ids(ledger, original_relation)
    cited_ids = set(candidate_audit.get("cited_evidence_ids", []))
    if not (candidate_ids & cited_ids):
        return False, "revision_citation_not_decision_grade", candidate_audit
    if (
        not same_label
        and original_ids
        and len(candidate_ids) <= len(original_ids)
    ):
        return False, "unresolved_opposing_decision_grade_evidence", candidate_audit

    if review:
        review_tokens = {
            token for token in re.findall(r"[a-z0-9]+", reason_text)
            if len(token) > 2
        }
        by_id = {item.get("id"): item for item in (ledger or [])}
        linked = False
        for item_id in cited_ids:
            item = by_id.get(item_id, {})
            if item.get("source") in {
                "targeted_region_verifier", "comparator"
            }:
                linked = True
                break
            evidence_tokens = {
                token for token in re.findall(
                    r"[a-z0-9]+", str(item.get("text", "")).casefold()
                ) if len(token) > 2
            }
            if len(review_tokens & evidence_tokens) >= 2:
                linked = True
                break
        if not linked:
            return False, "revision_not_linked_to_visual_review", candidate_audit

    accepted_ids = ",".join(sorted(candidate_ids & cited_ids))
    return (
        True,
        (
            "accepted_evidence_backed_confirmation:"
            if same_label
            else "accepted_stronger_current_image_evidence:"
        ) + accepted_ids,
        candidate_audit,
    )
