"""A reproducible decision record, not a generated post-hoc rationale."""
from copy import deepcopy
from engine.evidence_ledger import is_admissible_evidence


def final_artifact(caption, result):
    decision = result.get("decision", {}) or {}
    ledger = result.get("evidence_ledger", []) or []
    judge = result.get("judge", {}) or {}
    resolution = judge.get("tribunal_resolution", {}) or {}
    ids = list(dict.fromkeys(decision.get("_model_cited_evidence_ids", []) or []))
    cited = [deepcopy(item) for item in ledger
             if item.get("id") in ids and is_admissible_evidence(ledger, item)]
    valid_ids = {item["id"] for item in cited}
    by_id = {item.get("id"): item for item in ledger}
    ancestor_ids = set()
    pending = list(valid_ids)
    while pending:
        current = pending.pop()
        for parent in by_id.get(current, {}).get("derived_from_ids", []) or []:
            if parent not in ancestor_ids:
                ancestor_ids.add(parent)
                pending.append(parent)
    ancestors = [deepcopy(item) for item in ledger if item.get("id") in ancestor_ids
                 and is_admissible_evidence(ledger, item)]
    relation = {"ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"}.get(decision.get("label"))
    directional = [item["id"] for item in cited if item.get("relation") == relation
                   and item.get("grounded") and (item.get("decision_grade")
                   or (item.get("verification") or {}).get("decision_grade"))]
    changed = bool(resolution.get("accepted") and resolution.get("changed_decision"))
    origin = ("tribunal_revision" if changed else
              "control_baseline_fallback" if result.get("control", {}).get("baseline_fallback") else
              "control_vote" if result.get("control") else
              "tribunal_confirmation" if resolution.get("confirmation_valid") else
              "baseline_retained_after_review" if judge.get("requested") else
              "base_pipeline")
    reason = str(decision.get("explanation") or "")
    artifact = {
        "schema_version": "3.0", "source_caption": caption,
        "claim_graph": deepcopy((result.get("language_output") or {}).get("claim_graph")),
        "information_exchange": {
            "hearing_history": deepcopy((result.get("debate_details") or {}).get("hearing_history", [])),
            "current_witnesses": {role: deepcopy((result.get("debate_details") or {}).get(role, {}))
                                  for role in ("agent1_critique", "agent2_critique")},
            "review_transfers": [deepcopy(item.get("_communication_audit", {}))
                                 for item in judge.get("tribunal_reviews", [])],
            "semantic_correctness": "NOT_ESTABLISHED_BY_TRANSPORT_SUCCESS"},
        "claim_bindings": {key: deepcopy((result.get("language_output") or {}).get(key)) for key in (
            "claim_subject", "claim_predicate", "claim_object", "claim_source", "claim_target",
            "negation", "quantities", "claim_modifiers", "comparison_direction", "time_or_panel_scope")},
        "figurative_interpretation_hypothesis": (result.get("language_output") or {}).get("intended_meaning"),
        "candidate_dispute": deepcopy(result.get("candidate_dispute", {})),
        "control": deepcopy(result.get("control", {})),
        "semantic_bridge": deepcopy(resolution.get("semantic_bridge", {})),
        "semantic_bridge_verification": deepcopy(resolution.get("semantic_bridge_verification", {})),
        "citation_semantics_status": "model_checked_not_human_validated" if changed else "not_established_by_tribunal",
        "final_label": decision.get("label"), "decision_origin": origin,
        "confidence": decision.get("confidence"), "confidence_is_calibrated": False,
        "accepted_reason": reason, "cited_evidence": cited,
        "supporting_ancestors": ancestors,
        "directional_evidence_ids": directional,
        "invalid_or_inactive_citations": [key for key in ids if key not in valid_ids],
        "evidence_chain_complete": bool(directional) and len(valid_ids) == len(ids),
        "tribunal_execution_status": resolution.get("execution_status", "NOT_RUN"),
        "tribunal_gate_reason": resolution.get("reason", "not_requested"),
        "tribunal_acceptance_checks": deepcopy(resolution.get("acceptance_checks", [])),
        "alternatives": [{"proposal": item.get("provisional_verdict"),
                         "claim_node_resolution": deepcopy(item.get("claim_node_resolution")),
                         "node_relations": deepcopy(item.get("node_relations", [])),
                         "reason": item.get("reason"),
                         "counter_interpretation": item.get("counter_interpretation"),
                         "accepted": bool(resolution.get("accepted") or resolution.get("confirmation_valid")) and index == len(judge.get("tribunal_reviews", [])) - 1}
                        for index, item in enumerate(judge.get("tribunal_reviews", []))],
        "human_faithfulness_evaluated": False,
    }
    # The delivered trace is rendered from the record, not another free-form
    # explanation. Original model prose remains in accepted_reason for audit.
    evidence_text = " ".join(f"[{item['id']}] {item.get('text', '')}" for item in cited
                             if item["id"] in directional)
    artifact["delivered_explanation"] = (
        f"{decision.get('label')} ({origin}). "
        + (f"Recorded {relation} evidence: {evidence_text}" if evidence_text
           else "No accepted directional evidence chain was recorded."))
    if origin == "baseline_retained_after_review":
        artifact["delivered_explanation"] = (
            artifact["delivered_explanation"] + " "
            f"Tribunal did not replace this answer ({artifact['tribunal_gate_reason']}).")
    if not artifact["evidence_chain_complete"]:
        artifact["delivered_explanation"] += " The recorded citations do not establish a complete verified evidence chain."
    return artifact
