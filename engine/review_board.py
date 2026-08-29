"""Deterministic final review for all FigDebate revision proposals."""

import re

from engine.evidence_ledger import (
    audit_decision,
    evidence_provenance_roots,
    evidence_reliability,
)


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


def _grade_strength(ledger, relation, allowed_ids=None):
    """Combine independent provenance components, not repeated descriptions."""
    allowed = set(allowed_ids) if allowed_ids is not None else None
    candidates = []
    for item in ledger or []:
        if (
            not item.get("grounded", False)
            or item.get("relation") != relation
            or not _decision_grade(item)
            or (allowed is not None and item.get("id") not in allowed)
        ):
            continue
        roots = set(evidence_provenance_roots(ledger, item))
        if not roots and item.get("id"):
            roots = {item["id"]}
        candidates.append({
            "roots": roots,
            "strength": evidence_reliability(item),
        })
    # Any shared transitive source root places records in one component. This
    # prevents a witness statement, its cross-agent restatement, and the
    # tribunal's restatement from being counted as three independent facts.
    components = []
    for candidate in candidates:
        overlaps = [
            index for index, component in enumerate(components)
            if candidate["roots"] & component["roots"]
        ]
        if not overlaps:
            components.append(dict(candidate))
            continue
        merged_roots = set(candidate["roots"])
        merged_strength = candidate["strength"]
        for index in reversed(overlaps):
            component = components.pop(index)
            merged_roots.update(component["roots"])
            merged_strength = max(merged_strength, component["strength"])
        components.append({"roots": merged_roots, "strength": merged_strength})
    return round(sum(item["strength"] for item in components), 6)


def decision_grade_strength(ledger, relation, evidence_ids=None):
    return _grade_strength(ledger, relation, allowed_ids=evidence_ids)


def _mediated_tie_is_safe(
    candidate_label,
    candidate_ids,
    ledger,
    mediation,
    claim_contract,
):
    """Allow one narrow tie-break backed by new verified debate evidence."""
    plan = mediation or {}
    contract = claim_contract or {}
    if not plan.get("_usable", False):
        return False
    if plan.get("status") != "MEDIATE":
        return False
    if plan.get("provisional_verdict") != candidate_label:
        return False
    if float(plan.get("confidence") or 0.0) < 0.80:
        return False
    if plan.get("_invalid_evidence_ids"):
        return False
    if not contract.get("safe_for_directional_reasoning", False):
        return False

    by_id = {item.get("id"): item for item in (ledger or [])}
    cited = set(plan.get("_valid_evidence_ids", []) or [])
    if not any(
        by_id.get(item_id, {}).get("grounded", False)
        and by_id.get(item_id, {}).get("source") in {
            "agent1", "comparator", "targeted_region_verifier",
            "debate_visual_reinspection", "debate_visual_witness",
            "tribunal_independent_verifier",
            "cross_agent_relation_verifier",
            "tribunal_relation_verifier",
        }
        for item_id in cited
    ):
        return False

    return any(
        item_id in candidate_ids
        and by_id.get(item_id, {}).get("source") in {
            "targeted_region_verifier", "debate_visual_reinspection",
            "tribunal_independent_verifier",
            "cross_agent_relation_verifier",
            "tribunal_relation_verifier",
        }
        and _decision_grade(by_id.get(item_id, {}))
        for item_id in candidate_ids
    )


def audit_final_decision(decision, ledger, claim_contract=None):
    evidence = audit_decision(decision, ledger)
    contract = claim_contract or {}
    contract_valid = contract.get("safe_for_directional_reasoning")
    if (
        decision.get("decision_method") == "bounded_multimodal_tribunal"
        and contract.get("safe_for_tribunal_reasoning", False)
    ):
        contract_valid = True
    if decision.get("label") not in VALID_LABELS:
        status = "INVALID_BINARY_DECISION"
    elif evidence.get("valid"):
        status = "DIRECTIONALLY_GROUNDED"
    elif contract and not contract_valid:
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
            contract_valid
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
        if review["directionally_grounded"]:
            audit = review.get("evidence_audit", {}) or {}
            strength = _grade_strength(
                ledger,
                RELATION_FOR_LABEL.get(output.get("label")),
                allowed_ids=audit.get("cited_evidence_ids", []),
            )
            # Confidence is bounded by independently rooted verified evidence.
            evidence_cap = min(0.93, 0.55 + 0.40 * min(strength, 0.95))
            cap = min(cap if cap is not None else 1.0, evidence_cap)
            review["independent_evidence_strength"] = strength
            review["evidence_derived_confidence_cap"] = evidence_cap
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
    mediation=None,
):
    """Accept a change only when stronger current-image evidence supports it."""
    original_label = original.get("label")
    candidate_label = candidate.get("label")
    acceptance_checks = []

    def checked(name, passed, detail=""):
        acceptance_checks.append({
            "name": name,
            "passed": bool(passed),
            "detail": str(detail or ""),
        })
        return bool(passed)

    def reject(public_reason, audit=None, failed_invariant=None):
        diagnostics = dict(audit or {})
        diagnostics.update({
            "original_label": original_label,
            "candidate_label": candidate_label,
            "same_label": candidate_label == original_label,
            "accepted": False,
            "failed_invariant": failed_invariant or public_reason,
            "acceptance_checks": list(acceptance_checks),
        })
        return False, public_reason, diagnostics

    if candidate_label not in VALID_LABELS:
        checked("candidate_is_binary", False, candidate_label)
        return reject("invalid_proposed_label")
    checked("candidate_is_binary", True, candidate_label)
    same_label = candidate_label == original_label

    review = visual_review or {}
    recommendation = str(review.get("recommendation", "")).upper()
    if recommendation and recommendation not in {candidate_label, "ABSTAIN"}:
        checked("visual_reviewer_agrees_with_candidate", False, recommendation)
        return reject("visual_reviewer_recommends_other_label")
    checked("visual_reviewer_agrees_with_candidate", True, recommendation)
    if recommendation == "ABSTAIN":
        checked("visual_reviewer_is_directional", False, recommendation)
        return reject(
            "unchanged" if same_label else "visual_reviewer_abstained",
            failed_invariant="visual_reviewer_abstained",
        )
    checked("visual_reviewer_is_directional", True, recommendation)
    if review and not review.get("specific_evidence", False):
        checked("visual_review_has_specific_evidence", False)
        return reject("visual_review_lacks_specific_evidence")
    checked("visual_review_has_specific_evidence", True)
    reason_text = str(review.get("reason", "")).casefold()
    # Wording such as "does not support" may accompany affirmative opposing
    # evidence (for example, a visibly falling line). Missing evidence is kept
    # out of the revision path structurally: it is never decision grade, and a
    # candidate still has to cite a verified relation below.

    auditable_candidate = dict(candidate)
    if not auditable_candidate.get("_model_cited_evidence_ids"):
        auditable_candidate["_model_cited_evidence_ids"] = list(
            candidate.get("_evidence_audit", {}).get(
                "source_cited_evidence_ids", []
            )
        )
    candidate_audit = audit_decision(auditable_candidate, ledger)
    if not candidate_audit.get("source_valid"):
        checked("candidate_cites_current_image_source", False)
        return reject(
            "unchanged" if same_label
            else "revision_did_not_cite_current_image_evidence",
            candidate_audit,
            failed_invariant="revision_did_not_cite_current_image_evidence",
        )
    checked("candidate_cites_current_image_source", True)
    if not candidate_audit.get("valid"):
        checked("candidate_cites_decision_grade_direction", False)
        return reject(
            "unchanged" if same_label
            else "revision_lacks_decision_grade_direction",
            candidate_audit,
            failed_invariant="revision_lacks_decision_grade_direction",
        )
    checked("candidate_cites_decision_grade_direction", True)

    candidate_relation = RELATION_FOR_LABEL[candidate_label]
    original_relation = RELATION_FOR_LABEL.get(original_label)
    candidate_ids = _grade_ids(ledger, candidate_relation)
    original_ids = _grade_ids(ledger, original_relation)
    cited_ids = set(candidate_audit.get("cited_evidence_ids", []))
    candidate_strength = _grade_strength(
        ledger, candidate_relation, allowed_ids=cited_ids
    )
    original_strength = _grade_strength(ledger, original_relation)
    if not (candidate_ids & cited_ids):
        checked("citation_matches_candidate_direction", False)
        return reject("revision_citation_not_decision_grade", candidate_audit)
    checked("citation_matches_candidate_direction", True)

    # Agreement is useful diagnostic corroboration, but it is not a revision.
    # Keeping it outside the mutation path prevents a fallible judge from
    # raising confidence on an already-wrong label merely by repeating it.
    if same_label:
        candidate_audit["candidate_evidence_strength"] = _grade_strength(
            ledger, candidate_relation, allowed_ids=cited_ids
        )
        candidate_audit["original_evidence_strength"] = _grade_strength(
            ledger, original_relation
        )
        candidate_audit["confirmation_valid"] = True
        checked("proposal_changes_label", False, "same_label_confirmation")
        return reject(
            "same_label_confirmation_not_revision",
            candidate_audit,
            failed_invariant="proposal_does_not_change_label",
        )
    checked("proposal_changes_label", True)

    mediated_tie_break = False
    if (
        not same_label
        and original_ids
        and candidate_strength <= original_strength
    ):
        mediated_tie_break = _mediated_tie_is_safe(
            candidate_label,
            candidate_ids,
            ledger,
            mediation,
            claim_contract,
        )
        if not mediated_tie_break:
            checked(
                "candidate_is_stronger_or_safely_mediated", False,
                f"candidate={candidate_strength};original={original_strength}",
            )
            return reject(
                "unresolved_opposing_decision_grade_evidence",
                candidate_audit,
            )
    checked(
        "candidate_is_stronger_or_safely_mediated", True,
        f"candidate={candidate_strength};original={original_strength}",
    )

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
                "targeted_region_verifier", "tribunal_independent_verifier",
                "cross_agent_relation_verifier", "tribunal_relation_verifier",
                "comparator",
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
            checked("visual_review_links_to_cited_evidence", False)
            return reject("revision_not_linked_to_visual_review", candidate_audit)
        checked("visual_review_links_to_cited_evidence", True)

    accepted_ids = ",".join(sorted(candidate_ids & cited_ids))
    candidate_audit["candidate_evidence_strength"] = candidate_strength
    candidate_audit["original_evidence_strength"] = original_strength
    candidate_audit.update({
        "original_label": original_label,
        "candidate_label": candidate_label,
        "same_label": same_label,
        "accepted": True,
        "failed_invariant": "",
        "acceptance_checks": list(acceptance_checks),
    })
    return (
        True,
        (
            "accepted_mediated_verified_tiebreak:"
            if mediated_tie_break
            else "accepted_stronger_current_image_evidence:"
        ) + accepted_ids,
        candidate_audit,
    )
