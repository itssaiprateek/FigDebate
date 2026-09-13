"""Independent deterministic checks for semantic-bridge admissibility."""

from __future__ import annotations

import re

from engine.claim_contract import assess_tribunal_claim_dependencies
from engine.independent_review import audit_independent_record
from engine.evidence_ledger import evidence_provenance_roots, is_admissible_evidence
from engine.relation_semantics import is_missing_evidence_only
from engine.semantic_bridge import BRIDGE_FAMILIES, VISUAL_SOURCES


_SUPPORTIVE_REPRESENTATION = re.compile(
    r"\b(?:map(?:s|ped|ping)?|represent(?:s|ed|ing)?|depict(?:s|ed|ing)?|"
    r"illustrat(?:e|es|ed|ing)|exemplif(?:y|ies|ied)|visuali[sz](?:e|es|ed|ing)|"
    r"embod(?:y|ies|ied)|support(?:s|ed|ing)?|confirm(?:s|ed|ing)?)\b",
    re.I,
)
_EXPLICIT_OPPOSITION = re.compile(
    r"\b(?:opposite|opposes?|revers(?:e|es|ed|ing)|contradict(?:s|ed|ing|ory)?|"
    r"conflicts?|refut(?:e|es|ed|ing)|incompatible|instead|rather\s+than|"
    r"rises?\s+(?:while|but)\s+.*falls?|falls?\s+(?:while|but)\s+.*rises?)\b",
    re.I,
)
_EXPLICIT_ALIGNMENT = re.compile(
    r"\b(?:does\s+not\s+contradict|aligns?\s+with|consistent\s+with|"
    r"matches?|supports?|confirms?)\b",
    re.I,
)


def _ordered_relation_score(proposal, order):
    """Expose actual independent call votes, never the proposer's confidence."""
    calls = (proposal.get("independent_verification", {}) or {}).get("calls", [])
    call = next((item for item in calls if item.get("presentation_order") == list(order)), {})
    relation = call.get("relation", "UNRESOLVED")
    return relation, {label: int(relation == label) for label in ("SUPPORT", "CONFLICT")}


def verify_semantic_bridge(proposal, ledger, claim_contract, agent2_requirements_valid=True):
    """Return corroborated only when every mandatory provenance check passes."""
    proposal = dict(proposal or {})
    ledger = list(ledger or [])
    contract = claim_contract or {}
    by_id = {item.get("id"): item for item in ledger if item.get("id")}
    checks = []

    def checked(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail or "")})
        return bool(passed)

    visual_ids = list(proposal.get("visual_evidence_ids", []) or [])
    caption_ids = list(proposal.get("caption_evidence_ids", []) or [])
    visual_valid = bool(visual_ids) and all(
        item_id in by_id
        and is_admissible_evidence(ledger, by_id[item_id])
        and by_id[item_id].get("grounded", False)
        and by_id[item_id].get("source") in VISUAL_SOURCES
        for item_id in visual_ids
    )
    checked("grounded_visual_premise", visual_valid, ",".join(visual_ids))
    caption_valid = bool(caption_ids) and all(
        item_id in by_id
        and is_admissible_evidence(ledger, by_id[item_id])
        and by_id[item_id].get("source") == "agent2"
        and by_id[item_id].get("type") == "caption_proposition"
        for item_id in caption_ids
    )
    checked("immutable_caption_premise", caption_valid, ",".join(caption_ids))
    independent = audit_independent_record(proposal, ledger)
    checked("independent_semantic_verification", independent["valid"], independent)
    checked(
        "visual_premise_links_to_evidence",
        independent["bound_to_current_case"] and independent["premises_verified"],
    )
    checked(
        "caption_premise_preserved",
        independent["caption_verified"] and contract.get("proposition_preserved", False),
    )
    entity_ok = contract.get("entity_frame_preserved", False) and independent["entity_scope_verified"]
    checked("same_entity", entity_ok)
    family = str(proposal.get("bridge_type") or "").upper()
    declared_family = str(
        proposal.get("declared_bridge_type") or ""
    ).upper()
    checked("supported_bridge_family", family in BRIDGE_FAMILIES, family)
    # Incongruity explains why an item is humorous; it does not establish an
    # NLI direction. A directional family (polarity, comparison, temporal,
    # causal, and so on) must carry the actual support/conflict proof.
    checked(
        "directional_bridge_family",
        family != "HUMOR_INCONGRUITY" and not (
            declared_family == "HUMOR_INCONGRUITY"
            and family == "GENERAL_SEMANTIC_RELATION"
        ),
        f"derived={family};declared={declared_family}",
    )
    scope_ok = independent["entity_scope_verified"]
    checked("same_scope", scope_ok)
    relation = str(proposal.get("proposed_relation") or "").upper()
    checked("directional_relation", relation in {"SUPPORT", "CONFLICT"}, relation)
    bridge_statement = str(proposal.get("bridge_statement") or "")
    supportive_representation = bool(
        _SUPPORTIVE_REPRESENTATION.search(bridge_statement)
    )
    explicit_opposition = bool(_EXPLICIT_OPPOSITION.search(bridge_statement))
    # A statement that only says the image maps, depicts, or represents the
    # proposition cannot consistently be labelled CONFLICT. This catches
    # relation-polarity inversions without relying on a dataset phenomenon.
    statement_relation_consistent = not (
        relation == "CONFLICT"
        and (
            _EXPLICIT_ALIGNMENT.search(bridge_statement)
            or (supportive_representation and not explicit_opposition)
        )
    )
    argument_audit = (proposal.get("independent_verification") or {}).get("obligations", {}).get("arguments", {})
    version = (proposal.get("independent_verification") or {}).get("schema_version")
    factored = version in {"3.0", "4.0"}
    if version == "4.0":
        statement_relation_consistent = independent["bound_to_current_case"] and independent["arguments_verified"]
    elif factored:
        statement_relation_consistent = bool(independent["bound_to_current_case"] and argument_audit.get("_format_valid")
            and argument_audit.get("bridge_grounded") is True
            and argument_audit.get("bridge_relation") == relation)
    elif relation == "SUPPORT" and explicit_opposition and not _EXPLICIT_ALIGNMENT.search(bridge_statement):
        statement_relation_consistent = False
    checked(
        "bridge_statement_relation_consistent",
        statement_relation_consistent,
        relation,
    )
    missing_conflict = relation == "CONFLICT" and (
        is_missing_evidence_only(proposal.get("visual_premise"))
        or is_missing_evidence_only(proposal.get("bridge_statement"))
    )
    checked("conflict_is_affirmative", not missing_conflict)
    dependency_audit = assess_tribunal_claim_dependencies(
        contract,
        agent2_requirements_valid=agent2_requirements_valid,
        cited_evidence=[by_id[item_id] for item_id in visual_ids if item_id in by_id],
        proposed_relation=relation,
    )
    checked(
        "claim_dependencies_safe",
        dependency_audit["safe"],
        dependency_audit["reason"],
    )
    checked(
        "judge_admissibility_is_diagnostic_only",
        True,
        proposal.get("declared_admissibility"),
    )
    symmetric_present = bool(proposal.get("symmetric_cases_present", False))
    support_strength = conflict_strength = 0.0
    if symmetric_present:
        try:
            support_strength = float(proposal.get("support_strength"))
            conflict_strength = float(proposal.get("conflict_strength"))
        except (TypeError, ValueError):
            support_strength = conflict_strength = 0.0
        support_ids = set(proposal.get("support_evidence_ids", []) or [])
        conflict_ids = set(proposal.get("conflict_evidence_ids", []) or [])
        checked(
            "symmetric_case_evidence_valid",
            bool(support_ids or conflict_ids) and all(
                item_id in by_id and is_admissible_evidence(ledger, by_id[item_id])
                for item_id in support_ids | conflict_ids
            ),
        )
        winner = (
            "SUPPORT" if support_strength > conflict_strength
            else "CONFLICT" if conflict_strength > support_strength
            else "UNRESOLVED"
        )
        checked(
            "symmetric_case_direction_consistent",
            winner == relation,
            f"support={support_strength};conflict={conflict_strength}",
        )
        checked(
            "symmetric_case_margin",
            abs(support_strength - conflict_strength) >= 0.15,
            abs(support_strength - conflict_strength),
        )
    try:
        counter_strength = float(proposal.get("counter_interpretation_strength") or 0.0)
    except (TypeError, ValueError):
        counter_strength = 1.0
    if symmetric_present:
        counter_strength = max(
            counter_strength,
            conflict_strength if relation == "SUPPORT" else support_strength,
        )
    checked("counter_interpretation_resolved" if factored else "counter_interpretation_weak",
            independent["counter_resolved"] if factored else counter_strength <= 0.30,
            argument_audit if factored else counter_strength)
    try:
        confidence = float(proposal.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = 0.86 if contract.get("requires_normative_reasoning", False) else 0.82
    # Confidence is not proof. V3 replaces this self-rating veto with executed,
    # case-bound argument and premise checks; it never bypasses those checks.
    checked("verified_arguments" if factored else "bridge_confidence",
            independent["arguments_verified"] if factored else confidence >= threshold,
            argument_audit if factored else confidence)
    first, first_scores = _ordered_relation_score(proposal, ("SUPPORT", "CONFLICT"))
    reversed_choice, reversed_scores = _ordered_relation_score(
        proposal, ("CONFLICT", "SUPPORT")
    )
    if version == "4.0":
        checked("fresh_counterinterpretation_consistency", independent["arguments_verified"], independent.get("verification_control"))
    else:
        checked("position_reversed_consistency", first == reversed_choice == relation)
    visual_roots = {
        root for item_id in visual_ids for root in evidence_provenance_roots(ledger, item_id)
    }
    caption_roots = {
        root for item_id in caption_ids for root in evidence_provenance_roots(ledger, item_id)
    }
    checked("independent_provenance_classes", bool(visual_roots and caption_roots and not visual_roots & caption_roots))
    corroborated = bool(checks) and all(item["passed"] for item in checks)
    status = "BRIDGE_CORROBORATED" if corroborated else (
        "BRIDGE_PLAUSIBLE" if proposal.get("verification_status") != "INSUFFICIENT" else "INSUFFICIENT"
    )
    proposal.update({
        "same_entity_check": entity_ok,
        "same_scope_check": scope_ok,
        "verification_status": status,
        "provenance_roots": sorted(visual_roots | caption_roots),
    })
    return proposal, {
        "corroborated": corroborated,
        "verification_status": status,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "position_scores": {"forward": first_scores, "reversed": reversed_scores},
        "confidence_threshold": None if factored else threshold,
        "confidence_role": "diagnostic_only" if factored else "legacy_threshold",
        "confidence_is_calibrated": False,
        "independent_verification": independent,
    }
