"""Bounded tribunal state and deterministic resolution acceptance."""

from copy import deepcopy

from engine.decision_trace import (
    append_decision_checkpoint,
    attach_decision_trace,
)
from engine.evidence_ledger import (
    add_tribunal_corroborated_relation,
    attach_evidence_audit,
)
from engine.review_board import attach_final_review, review_revision


TRIBUNAL_SCHEMA_VERSION = "1.0"
MAX_TRIBUNAL_ROUNDS = 2
MIN_TRIBUNAL_RESOLUTION_CONFIDENCE = 0.75
RELATION_FOR_LABEL = {"ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"}


def new_tribunal_session(initial_plan=None):
    return {
        "schema_version": TRIBUNAL_SCHEMA_VERSION,
        "state": "QUESTIONS_ISSUED" if initial_plan else "OPEN",
        "max_rounds": MAX_TRIBUNAL_ROUNDS,
        "rounds": [],
        "initial_plan": deepcopy(initial_plan or {}),
        "stop_reason": "",
    }


def record_tribunal_round(session, review, debate_details=None):
    output = deepcopy(session or new_tribunal_session())
    round_number = len(output["rounds"]) + 1
    output["rounds"].append({
        "round": round_number,
        "review": deepcopy(review or {}),
        "agent1_response": deepcopy(
            (debate_details or {}).get("agent1_critique", {})
        ),
        "agent2_response": deepcopy(
            (debate_details or {}).get("agent2_critique", {})
        ),
    })
    status = (review or {}).get("status")
    if status == "FOLLOW_UP" and round_number < output["max_rounds"]:
        output["state"] = "FOLLOW_UP_REQUIRED"
    elif status == "RESOLVE":
        output["state"] = "READY_FOR_ARBITRATION"
    else:
        output["state"] = "ABSTAINED"
        output["stop_reason"] = (
            "maximum_rounds_reached" if status == "FOLLOW_UP"
            else "mediator_abstained"
        )
    return output


def followup_plan(review):
    if (review or {}).get("status") != "FOLLOW_UP":
        return {}
    return {
        "status": "MEDIATE",
        "provisional_verdict": "ABSTAIN",
        "confidence": review.get("confidence", 0.0),
        "agent1_questions": list(review.get("agent1_questions", []) or [])[:1],
        "agent2_questions": list(review.get("agent2_questions", []) or [])[:1],
        "verification_requests": list(
            review.get("verification_requests", []) or []
        )[:1],
        "disputed_issues": [review.get("issue", "")],
        "_valid_evidence_ids": list(
            review.get("_valid_evidence_ids", []) or []
        ),
        "_invalid_evidence_ids": list(
            review.get("_invalid_evidence_ids", []) or []
        ),
        "_format_valid": bool(review.get("_format_valid", False)),
        "_usable": bool(
            review.get("_format_valid", False)
            and not review.get("_invalid_evidence_ids")
            and (
                review.get("agent1_questions")
                or review.get("agent2_questions")
            )
        ),
    }


def apply_tribunal_resolution(
    current_decision, review, ledger, claim_contract=None,
    agent2_requirements_valid=True, agent1_critique=None,
    agent2_critique=None,
):
    """Promote corroborated evidence and apply the ordinary review board."""
    current_decision = current_decision or {}
    review = review or {}
    ledger = ledger or []
    metadata = {
        "accepted": False,
        "changed_decision": False,
        "previous_label": current_decision.get("label"),
        "proposed_label": review.get("provisional_verdict"),
        "reason": "",
        "acceptance_checks": [],
    }

    def checked(name, passed, detail=""):
        metadata["acceptance_checks"].append({
            "name": name,
            "passed": bool(passed),
            "detail": str(detail or ""),
        })
        return bool(passed)

    def reject(reason, output_ledger=None):
        metadata["reason"] = reason
        trace = append_decision_checkpoint(
            current_decision.get("_decision_trace", []),
            "tribunal_resolution_preserved",
            current_decision,
            ledger=(output_ledger if output_ledger is not None else ledger),
            metadata={
                "reason": reason,
                "status": review.get("status"),
                "proposed_label": review.get("provisional_verdict"),
            },
        )
        return (
            attach_decision_trace(current_decision, trace),
            list(output_ledger if output_ledger is not None else ledger),
            metadata,
        )

    try:
        review_confidence = float(review.get("confidence") or 0.0)
    except (TypeError, ValueError):
        review_confidence = 0.0
    contract = claim_contract or {}
    directional_frame_is_defined = bool(
        contract.get("relation_pair_valid", False)
    )
    checks = (
        (review.get("_format_valid", False), "invalid_tribunal_contract"),
        (review.get("status") == "RESOLVE", "tribunal_not_resolved"),
        (review.get("provisional_verdict") in RELATION_FOR_LABEL,
         "tribunal_has_no_binary_verdict"),
        (review_confidence >= MIN_TRIBUNAL_RESOLUTION_CONFIDENCE,
         "tribunal_confidence_below_threshold"),
        (not review.get("_invalid_evidence_ids"),
         "tribunal_cited_unknown_evidence"),
        (bool(review.get("_valid_evidence_ids")),
         "tribunal_cited_no_current_image_evidence"),
        (bool(review.get("visual_observations")),
         "tribunal_reported_no_visual_observation"),
        (bool((claim_contract or {}).get("safe_for_directional_reasoning", False)),
         "claim_contract_not_safe"),
        (directional_frame_is_defined, "claim_direction_not_well_defined"),
        (bool(agent2_requirements_valid), "agent2_requirements_invalid"),
    )
    for passed, reason in checks:
        checked(reason, passed)
        if not passed:
            return reject(reason)

    verified_ledger, corroboration = add_tribunal_corroborated_relation(
        ledger, review, claim_contract, agent1_critique, agent2_critique,
    )
    metadata["corroboration"] = corroboration
    by_id = {item.get("id"): item for item in verified_ledger}
    cited_ids = list(review.get("_valid_evidence_ids", []) or [])
    if corroboration.get("promoted", False):
        cited_ids.append(corroboration["evidence_id"])
    citations_grounded = any(
        by_id.get(item_id, {}).get("grounded", False)
        and by_id.get(item_id, {}).get("source") in {
            "agent1", "comparator", "debate_visual_witness",
            "targeted_region_verifier", "debate_visual_reinspection",
            "cross_agent_relation_verifier",
            "tribunal_relation_verifier",
        }
        for item_id in cited_ids
    )
    checked("tribunal_citations_grounded", citations_grounded)
    if not citations_grounded:
        return reject("tribunal_citations_not_grounded")

    proposed_label = review["provisional_verdict"]
    relation = RELATION_FOR_LABEL[proposed_label]
    # A mediator may select and explain existing proof, but its prose alone is
    # never evidence. Resolution uses either an existing verified relation or
    # the three-source relation constructed above from a current visual
    # witness, a preserved claim audit, and the independent tribunal relation.
    verified_ids = [
        item_id for item_id in cited_ids
        if by_id.get(item_id, {}).get("grounded", False)
        and by_id.get(item_id, {}).get("relation") == relation
        and (
            by_id.get(item_id, {}).get("decision_grade", False)
            or by_id.get(item_id, {}).get("verification", {}).get(
                "decision_grade", False
            )
        )
    ]
    checked(
        "tribunal_citations_match_decision_grade_direction",
        bool(verified_ids),
        ",".join(verified_ids),
    )
    if not verified_ids:
        return reject("tribunal_citations_lack_independent_direction")
    observation_text = " ".join(review.get("visual_observations", []))
    verified_ledger = list(verified_ledger)
    candidate = dict(current_decision)
    candidate.update({
        "label": proposed_label,
        "confidence": min(review_confidence, 0.85),
        "decision_method": "bounded_multimodal_tribunal",
        "explanation": review.get("reason", ""),
        "_model_cited_evidence_ids": verified_ids,
        "_final_decision_valid": True,
        "_raw_tribunal_proposed_label": proposed_label,
    })
    accepted, reason, candidate_audit = review_revision(
        current_decision,
        candidate,
        verified_ledger,
        visual_review={
            "recommendation": proposed_label,
            "specific_evidence": True,
            "reason": observation_text + " " + review.get("reason", ""),
        },
        claim_contract=claim_contract,
    )
    metadata["reason"] = reason
    metadata["candidate_evidence_audit"] = candidate_audit
    metadata["verified_evidence_id"] = verified_ids[0]
    checked(
        "deterministic_review_board_acceptance",
        accepted,
        candidate_audit.get("failed_invariant", reason),
    )
    if not accepted:
        return reject(reason, verified_ledger)

    candidate = attach_evidence_audit(candidate, verified_ledger)
    candidate = attach_final_review(candidate, verified_ledger, claim_contract)
    trace = append_decision_checkpoint(
        current_decision.get("_decision_trace", []),
        "tribunal_resolution_accepted",
        candidate,
        ledger=verified_ledger,
        metadata={"reason": reason},
    )
    candidate = attach_decision_trace(candidate, trace)
    metadata["accepted"] = True
    metadata["changed_decision"] = (
        current_decision.get("label") != candidate.get("label")
    )
    return candidate, verified_ledger, metadata
