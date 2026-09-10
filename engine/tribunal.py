"""Bounded tribunal state and deterministic resolution acceptance."""

from copy import deepcopy
from engine.review_outcome import classify_review

from engine.claim_contract import assess_tribunal_claim_dependencies
from engine.decision_trace import (
    append_decision_checkpoint,
    attach_decision_trace,
)
from engine.evidence_ledger import (
    add_semantic_bridge_evidence,
    add_tribunal_corroborated_relation,
    attach_evidence_audit,
)
from engine.review_board import attach_final_review, review_revision
from engine.relation_semantics import is_missing_evidence_only
from engine.semantic_bridge import build_semantic_bridge
from engine.semantic_bridge_verifier import verify_semantic_bridge


TRIBUNAL_SCHEMA_VERSION = "2.0"
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
    outcome = classify_review(review)
    if outcome["execution_status"] == "FAILED":
        output.update(state="EXECUTION_FAILED", stop_reason=outcome["execution_error_type"])
        return output
    if outcome["context_status"] not in {"VALID", "NOT_CHECKED"}:
        output.update(state=outcome["context_status"], stop_reason="tribunal_context:" + outcome["context_status"])
        return output
    if outcome["schema_status"] != "VALID":
        output.update(state="INVALID_OUTPUT", stop_reason="invalid_tribunal_contract")
        return output
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


def followup_plan(review, comparison=None):
    if (review or {}).get("status") != "FOLLOW_UP":
        return {}
    agent1_questions = list(review.get("agent1_questions", []) or [])
    agent2_questions = list(review.get("agent2_questions", []) or [])
    verification_requests = list(review.get("verification_requests", []) or [])
    if review.get("requested_follow_up") and not (
        agent1_questions or agent2_questions
    ):
        from engine.question_router import build_question_plan
        deterministic = build_question_plan(comparison or {}).to_dict()
        requested = review.get("requested_follow_up")
        if requested in {"VISUAL_PREMISE", "ENTITY_BINDING", "SCOPE_BINDING"}:
            agent1_questions = [deterministic["agent1_question"]]
        elif requested == "CAPTION_PREMISE":
            agent2_questions = [deterministic["agent2_question"]]
        else:
            verification_requests = [deterministic["verification_request"]]
    return {
        "status": "MEDIATE",
        "provisional_verdict": "ABSTAIN",
        "confidence": review.get("confidence", 0.0),
        "agent1_questions": agent1_questions,
        "agent2_questions": agent2_questions,
        "verification_requests": verification_requests,
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
                agent1_questions or agent2_questions or verification_requests
            )
        ),
    }


def repair_followup_plan(review, comparison=None, debate_details=None):
    """Route repairable resolution defects into one label-blind hearing."""
    review = review or {}
    debate = debate_details or {}
    agent2 = debate.get("agent2_critique", {}) or {}
    reasons = []
    verification = review.get("_independent_verification") or {}
    if verification.get("schema_version") == "3.0":
        obligations = verification.get("obligations") or {}
        # Route actual failed obligations; self-rated confidence/strength is
        # not a substitute for a verification result in the factored path.
        for key, issue in (("visual", "VISUAL_PREMISE"), ("caption", "CAPTION_PREMISE"),
                           ("mapping", "ENTITY_SCOPE_MAPPING")):
            if obligations.get(key, {}).get("verified") is not True:
                reasons.append(issue)
        args = obligations.get("arguments") or {}
        if args.get("bridge_grounded") is not True:
            reasons.append("BRIDGE_GROUNDING")
        if args.get("bridge_relation") != review.get("relation"):
            reasons.append("BRIDGE_DIRECTION")
        calls = verification.get("calls") or []
        if len(calls) == 2 and calls[0].get("relation") != calls[1].get("relation"):
            reasons.append("RELATION_DISAGREEMENT")
        if (args.get("counter_resolved") is not True and review.get("counter_interpretation")
                and args.get("counter_relation") != review.get("relation")):
            reasons.append("COUNTER_INTERPRETATION")
        if not reasons:
            return {}
        visual_question = ("Reinspect the current image. Report the exact visible text, entity or panel "
            "attachments and observed outcomes; separate visible facts from intentions or missing evidence.")
        language_question = ("Using only the original caption, identify the complete expressed proposition "
            "and licensed idiomatic readings with exact source quotes. Preserve roles, negation, degree "
            "and scope; distinguish sarcastic intention from what is asserted. State unresolved ambiguity.")
        return {"status": "MEDIATE", "provisional_verdict": "ABSTAIN", "confidence": 0.0,
            "agent1_questions": [visual_question] if any(r in reasons for r in
                ("VISUAL_PREMISE", "ENTITY_SCOPE_MAPPING", "COUNTER_INTERPRETATION", "BRIDGE_GROUNDING", "BRIDGE_DIRECTION", "RELATION_DISAGREEMENT")) else [],
            "agent2_questions": [language_question] if any(r in reasons for r in
                ("CAPTION_PREMISE", "ENTITY_SCOPE_MAPPING", "COUNTER_INTERPRETATION", "BRIDGE_DIRECTION", "RELATION_DISAGREEMENT")) else [],
            "verification_requests": [], "disputed_issues": reasons,
            "_valid_evidence_ids": list(review.get("_valid_evidence_ids", []) or []),
            "_invalid_evidence_ids": [], "_format_valid": True, "_usable": True,
            "repair_reasons": reasons, "origin": "failed_factored_obligations",
            "repair_subject": {"visual_premise": review.get("visual_premise", ""),
                "witness_visibility": "withheld_for_fresh_caption_blind_observation"}}
    if not agent2.get("requirements_valid", True):
        reasons.append("AGENT2_DIRECTIONAL_CONTRACT")
    if str(review.get("semantic_bridge_type") or "").upper() == "HUMOR_INCONGRUITY":
        reasons.append("FIGURATIVE_DIRECTION")
    try:
        if float(review.get("counter_interpretation_strength") or 0.0) > 0.30:
            reasons.append("COUNTER_INTERPRETATION")
    except (TypeError, ValueError):
        reasons.append("COUNTER_INTERPRETATION")
    if str(review.get("admissibility") or "").upper() in {
        "PLAUSIBLE", "INSUFFICIENT",
    }:
        reasons.append("PREMISE_VERIFICATION")
    if not reasons:
        return {}

    from engine.question_router import build_question_plan

    routed = build_question_plan(comparison or {}).to_dict()
    agent1_question = routed.get("agent1_question", "")
    agent2_question = routed.get("agent2_question", "")
    if "AGENT2_DIRECTIONAL_CONTRACT" in reasons:
        agent2_question = (
            "Restate the preserved proposition, expected state, and affirmative "
            "opposite state with identical entities and scope."
        )
    if "FIGURATIVE_DIRECTION" in reasons:
        agent1_question = (
            "State the visible expectation and observed outcome separately; "
            "do not decide whether the caption is true."
        )
        agent2_question = (
            "State whether the joke asserts, reverses, or merely illustrates "
            "the preserved proposition, and identify the exact cue."
        )
    if "COUNTER_INTERPRETATION" in reasons:
        agent2_question += (
            " Give the strongest genuinely opposing interpretation and the "
            "observable fact that distinguishes it."
        )
    return {
        "status": "MEDIATE",
        "provisional_verdict": "ABSTAIN",
        "confidence": 0.5,
        "agent1_questions": [agent1_question] if agent1_question else [],
        "agent2_questions": [agent2_question] if agent2_question else [],
        "verification_requests": [routed.get("verification_request", "")],
        "disputed_issues": sorted(set(reasons)),
        "_valid_evidence_ids": list(review.get("_valid_evidence_ids", []) or []),
        "_invalid_evidence_ids": [],
        "_format_valid": True,
        "_usable": bool(agent1_question or agent2_question),
        "repair_reasons": sorted(set(reasons)),
    }


def apply_tribunal_resolution(
    current_decision, review, ledger, claim_contract=None,
    agent2_requirements_valid=True, agent1_critique=None,
    agent2_critique=None,
    semantic_bridge_mode="disabled", language_output=None,
):
    """Promote corroborated evidence and apply the ordinary review board."""
    current_decision = current_decision or {}
    review = review or {}
    ledger = ledger or []
    resolution_ledger = list(ledger)
    metadata = {
        "accepted": False,
        "changed_decision": False,
        "previous_label": current_decision.get("label"),
        "previous_confidence": current_decision.get("confidence"),
        "proposed_label": review.get("provisional_verdict"),
        "reason": "",
        "acceptance_checks": [],
        "semantic_judgment": review.get(
            "best_semantic_judgment", review.get("provisional_verdict")
        ),
        "semantic_judgment_valid": review.get(
            "best_semantic_judgment", review.get("provisional_verdict")
        ) in RELATION_FOR_LABEL,
        "semantic_abstained": review.get(
            "best_semantic_judgment", review.get("provisional_verdict")
        ) == "ABSTAIN",
        "admissibility": "PENDING",
        "contract_normalizations": list(
            review.get("_normalized_fields", []) or []
        ),
    }

    metadata.update(classify_review(review))

    def checked(name, passed, detail=""):
        metadata["acceptance_checks"].append({
            "name": name,
            "passed": bool(passed),
            "detail": str(detail or ""),
        })
        return bool(passed)

    def reject(reason, output_ledger=None):
        metadata["reason"] = reason
        metadata["admissibility"] = "REJECTED"
        trace = append_decision_checkpoint(
            current_decision.get("_decision_trace", []),
            "tribunal_resolution_preserved",
            current_decision,
            ledger=(output_ledger if output_ledger is not None else resolution_ledger),
            metadata={
                "reason": reason,
                "status": review.get("status"),
                "proposed_label": review.get("provisional_verdict"),
            },
        )
        return (
            attach_decision_trace(current_decision, trace),
            list(output_ledger if output_ledger is not None else resolution_ledger),
            metadata,
        )

    if metadata["execution_status"] == "FAILED":
        return reject("tribunal_execution_failed")
    if metadata["schema_status"] != "VALID":
        return reject("invalid_tribunal_contract")
    if review.get("_context_valid") is False:
        return reject("tribunal_context:" + review.get("_context_status", "INVALID"))

    try:
        review_confidence = float(review.get("confidence") or 0.0)
    except (TypeError, ValueError):
        review_confidence = 0.0
    contract = claim_contract or {}
    if semantic_bridge_mode not in {"disabled", "shadow", "corroborated"}:
        raise ValueError(f"Unknown semantic bridge mode: {semantic_bridge_mode}")
    if semantic_bridge_mode != "disabled":
        bridge = build_semantic_bridge(
            review, resolution_ledger, contract, language_output
        )
        verified_bridge, bridge_verification = verify_semantic_bridge(
            bridge,
            resolution_ledger,
            contract,
            agent2_requirements_valid=agent2_requirements_valid,
        )
        if semantic_bridge_mode == "shadow":
            diagnostic_ledger, bridge_record = add_semantic_bridge_evidence(
                resolution_ledger, verified_bridge, bridge_verification
            )
            metadata["shadow_evidence"] = diagnostic_ledger[len(resolution_ledger):]
            bridge_record = dict(bridge_record, diagnostic_only=True, promoted=False)
        else:
            resolution_ledger, bridge_record = add_semantic_bridge_evidence(
                resolution_ledger, verified_bridge, bridge_verification
            )
        metadata["semantic_bridge"] = verified_bridge
        metadata["semantic_bridge_verification"] = bridge_verification
        metadata["semantic_bridge_record"] = bridge_record
        metadata["semantic_bridge_mode"] = semantic_bridge_mode
        if semantic_bridge_mode == "corroborated" and bridge_record.get("promoted"):
            review = dict(review)
            bridge_relation = verified_bridge["proposed_relation"]
            review.update({
                "status": "RESOLVE",
                "relation": bridge_relation,
                "provisional_verdict": (
                    "ENTAILS" if bridge_relation == "SUPPORT" else "CONTRADICTS"
                ),
                "visual_observations": [verified_bridge["visual_premise"]],
                "_valid_evidence_ids": list(dict.fromkeys(
                    list(review.get("_valid_evidence_ids", []) or [])
                    + [bridge_record["evidence_id"]]
                )),
                "_invalid_evidence_ids": [],
            })
            metadata["proposed_label"] = review["provisional_verdict"]
            metadata["semantic_bridge_applied_to_candidate"] = True
    proposed_label = review.get("provisional_verdict")
    proposed_relation = RELATION_FOR_LABEL.get(proposed_label)
    current_by_id = {
        item.get("id"): item for item in resolution_ledger if item.get("id")
    }
    cited_items = [
        current_by_id[item_id]
        for item_id in (review.get("_valid_evidence_ids", []) or [])
        if item_id in current_by_id
    ]
    dependency_audit = assess_tribunal_claim_dependencies(
        contract,
        agent2_requirements_valid=agent2_requirements_valid,
        cited_evidence=cited_items,
        proposed_relation=proposed_relation,
    )
    metadata["claim_dependency_audit"] = dependency_audit
    checks = (
        (review.get("_format_valid", False), "invalid_tribunal_contract"),
        (review.get("status") == "RESOLVE", "tribunal_not_resolved"),
        (review.get("provisional_verdict") in RELATION_FOR_LABEL,
         "tribunal_has_no_binary_verdict"),
        (bool(metadata.get("semantic_bridge_applied_to_candidate")) or review_confidence >= MIN_TRIBUNAL_RESOLUTION_CONFIDENCE,
         "tribunal_confidence_below_threshold"),
        (not review.get("_invalid_evidence_ids"),
         "tribunal_cited_unknown_evidence"),
        (bool(review.get("_valid_evidence_ids")),
         "tribunal_cited_no_current_image_evidence"),
        (bool(review.get("visual_observations")),
         "tribunal_reported_no_visual_observation"),
        (dependency_audit["immutable_safe"],
         "immutable_caption_dependencies_invalid"),
        (dependency_audit["relation_safe"],
         "directional_claim_dependencies_invalid"),
    )
    for passed, reason in checks:
        checked(reason, passed)
        if not passed:
            return reject(reason)

    verified_ledger, corroboration = add_tribunal_corroborated_relation(
        resolution_ledger, review, claim_contract, agent1_critique, agent2_critique,
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
            "semantic_bridge_verifier",
        }
        for item_id in cited_ids
    )
    checked("tribunal_citations_grounded", citations_grounded)
    if not citations_grounded:
        return reject("tribunal_citations_not_grounded")

    proposed_label = review["provisional_verdict"]
    if proposed_label == current_decision.get("label"):
        metadata["same_label_agreement"] = True
        metadata["confirmation_valid"] = any(
            by_id.get(key, {}).get("grounded")
            and by_id.get(key, {}).get("relation") == RELATION_FOR_LABEL[proposed_label]
            and (by_id.get(key, {}).get("decision_grade")
                 or (by_id.get(key, {}).get("verification") or {}).get("decision_grade"))
            for key in cited_ids)
        if metadata["confirmation_valid"]:
            current_decision = deepcopy(current_decision)
            current_decision["_model_cited_evidence_ids"] = cited_ids
            current_decision = attach_evidence_audit(current_decision, verified_ledger)
            current_decision = attach_final_review(current_decision, verified_ledger, contract)
        return reject("same_label_confirmation_not_revision" if metadata["confirmation_valid"]
                      else "same_label_unverified_preserved", verified_ledger)
    relation = RELATION_FOR_LABEL[proposed_label]
    bridge_family = str(
        (metadata.get("semantic_bridge") or {}).get("bridge_type") or ""
    ).upper()
    declared_bridge_family = str(
        (metadata.get("semantic_bridge") or {}).get("declared_bridge_type") or ""
    ).upper()
    bridge_corroborated = bool(
        (metadata.get("semantic_bridge_verification") or {}).get(
            "corroborated", False
        )
    )
    # Humor/incongruity is a trigger for closer inspection, not proof that the
    # caption is false. A conflict revision needs a directional bridge family
    # such as polarity reversal, comparison direction, temporal order, or
    # participant outcome. This applies even when another derived tribunal
    # record was marked decision-grade from the same interpretation.
    if relation == "CONFLICT" and (
        bridge_family == "HUMOR_INCONGRUITY"
        or (
            declared_bridge_family == "HUMOR_INCONGRUITY"
            and not bridge_corroborated
        )
    ):
        checked(
            "figurative_incongruity_is_not_directional_conflict",
            False,
            declared_bridge_family or bridge_family,
        )
        return reject(
            "figurative_incongruity_requires_directional_relation_proof",
            verified_ledger,
        )
    checked(
        "figurative_incongruity_is_not_directional_conflict",
        True,
        bridge_family,
    )
    if (
        relation == "CONFLICT"
        and contract.get("requires_normative_reasoning", False)
        and not bridge_corroborated
    ):
        checked(
            "normative_evaluation_has_verified_factual_opposite",
            False,
            "charitable_intent_or_missing_context_is_not_conflict",
        )
        return reject(
            "normative_evaluation_requires_verified_factual_opposite",
            verified_ledger,
        )
    checked(
        "normative_evaluation_has_verified_factual_opposite",
        True,
        "not_applicable_or_verified",
    )
    if relation == "CONFLICT" and (
        is_missing_evidence_only(review.get("reason"))
        or all(
            is_missing_evidence_only(item)
            for item in (review.get("visual_observations", []) or [])
        )
    ):
        return reject("conflict_based_only_on_missing_evidence", verified_ledger)
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
    metadata["admissibility"] = "ACCEPTED"
    metadata["changed_decision"] = (
        current_decision.get("label") != candidate.get("label")
    )
    return candidate, verified_ledger, metadata
