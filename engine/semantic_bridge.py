"""Auditable semantic bridge proposals between visual and caption premises."""

from __future__ import annotations

import re

from engine.evidence_ledger import is_admissible_evidence
from engine.reasoning_schema import normalize_structural_type
from engine.relation_semantics import is_missing_evidence_only


BRIDGE_FAMILIES = {
    "AFFECTIVE_OPPOSITION",
    "LITERAL_INTENDED_POLARITY",
    "TEMPORAL_SEQUENCE",
    "CAUSE_EFFECT",
    "PARTICIPANT_ACTION_OUTCOME",
    "COMPARISON_DIRECTION",
    "QUOTED_STATEMENT_REACTION",
    "SYMBOL_TARGET_ATTACHMENT",
    "OBJECT_FUNCTION",
    "SOCIAL_CONVENTION",
    "HUMOR_INCONGRUITY",
    "GENERAL_SEMANTIC_RELATION",
}
BRIDGE_LEVELS = {
    "DIRECT_VERIFIED", "BRIDGE_CORROBORATED", "BRIDGE_PLAUSIBLE", "INSUFFICIENT",
}
VISUAL_SOURCES = {
    "agent1", "debate_visual_witness", "targeted_region_verifier",
    "debate_visual_reinspection", "structured_observation_builder",
}


def _clean(value):
    # Identity and interpretation checks need the complete text. Inference
    # context limits are checked at the model boundary, never by silent slicing.
    return " ".join(str(value or "").split())


def select_bridge_family(claim_contract, language_output=None):
    """Select from structure only; dataset phenomenon labels are never inputs."""
    contract = claim_contract or {}
    language = language_output or {}
    structural = normalize_structural_type(
        contract.get("structural_reasoning_type")
        or language.get("structural_reasoning_type")
    )
    mechanisms = {
        str(item or "").strip().upper()
        for item in contract.get("figurative_mechanism_candidates", []) or []
    }
    relation_family = str(language.get("relation_family") or "").casefold()
    mapping = {
        "AFFECTIVE_SCENE": "AFFECTIVE_OPPOSITION",
        "TEMPORAL_CAUSAL_SEQUENCE": "TEMPORAL_SEQUENCE",
        "QUOTED_STATEMENT_AND_REACTION": "QUOTED_STATEMENT_REACTION",
        "COMPARATIVE_LAYOUT": "COMPARISON_DIRECTION",
        "OCR_REGION_BINDING": "QUOTED_STATEMENT_REACTION",
    }
    if structural in mapping:
        return mapping[structural]
    if "SARCASM_POLARITY" in mechanisms:
        return "LITERAL_INTENDED_POLARITY"
    if "METAPHOR_MAPPING" in mechanisms:
        return "SYMBOL_TARGET_ATTACHMENT"
    if any("HUMOR" in item or "INCONGRU" in item for item in mechanisms):
        return "HUMOR_INCONGRUITY"
    if relation_family in {"outcome", "pace", "quantity", "trajectory"}:
        return "PARTICIPANT_ACTION_OUTCOME"
    if relation_family in {"sentiment", "safety", "trust"}:
        return "AFFECTIVE_OPPOSITION"
    return "GENERAL_SEMANTIC_RELATION"


def build_semantic_bridge(review, ledger, claim_contract, language_output=None):
    """Build a non-authoritative proposal with explicit existing premises."""
    review = review or {}
    ledger = list(ledger or [])
    contract = claim_contract or {}
    by_id = {item.get("id"): item for item in ledger if item.get("id")}
    cited = list(review.get("_valid_evidence_ids") or review.get("evidence_ids") or [])
    visual_ids = [
        item_id for item_id in cited
        if item_id in by_id
        and is_admissible_evidence(ledger, by_id[item_id])
        and by_id[item_id].get("grounded", False)
        and by_id[item_id].get("source") in VISUAL_SOURCES
    ]
    caption_ids = [
        item.get("id") for item in ledger
        if item.get("source") == "agent2"
        and item.get("type") == "caption_proposition"
        and item.get("id")
    ][:1]
    source_caption = (contract.get("source_caption") or contract.get("caption_proposition")
                      or (by_id[caption_ids[0]].get("text", "") if caption_ids else ""))
    visual_premise = _clean(
        review.get("visual_premise")
        or " ".join(review.get("visual_observations", []) or [])
        or " ".join(by_id[item_id].get("text", "") for item_id in visual_ids)
    )
    caption_premise = _clean(
        review.get("caption_premise") or source_caption
    )
    declared_family = str(
        review.get("semantic_bridge_type") or ""
    ).strip().upper()
    structured_family = select_bridge_family(contract, language_output)
    # The bridge family is a deterministic property of the typed claim.  A
    # judge may describe a relation, but cannot choose the verification rules
    # that will later decide whether its own proposal is admissible.
    family = structured_family
    relation = str(review.get("relation") or "UNRESOLVED").strip().upper()
    if relation not in {"SUPPORT", "CONFLICT", "UNRESOLVED"}:
        relation = "UNRESOLVED"
    bridge_statement = _clean(review.get("semantic_bridge") or review.get("reason"))
    counter = _clean(review.get("counter_interpretation"))
    try:
        confidence = max(0.0, min(float(review.get("confidence") or 0.0), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    level = "BRIDGE_PLAUSIBLE"
    deficiencies = []
    if not visual_ids or not visual_premise:
        deficiencies.append("missing_grounded_visual_premise")
    if not caption_ids or not caption_premise:
        deficiencies.append("missing_caption_premise")
    if relation == "UNRESOLVED":
        deficiencies.append("unresolved_relation")
    if relation == "CONFLICT" and (
        is_missing_evidence_only(visual_premise)
        or is_missing_evidence_only(bridge_statement)
    ):
        deficiencies.append("missing_evidence_cannot_create_conflict")
    if deficiencies:
        level = "INSUFFICIENT"
    return {
        "schema_version": "2.0",
        "claim_frame": {key: contract.get(key) for key in (
            "claim_subject", "claim_predicate", "claim_object", "claim_source",
            "claim_target", "literal_polarity", "comparison_direction", "time_or_panel_scope")}
            if not contract.get("source_identity_graph_valid") else
            {"source_caption": source_caption, "representation": "UNDECOMPOSED_SOURCE"},
        "source_caption": source_caption,
        "interpreted_assertion": (review.get('_compact_value') or {}).get('interpreted_assertion', ''),
        "image_sha256": review.get("_case_image_sha256", ""),
        "verification_task": review.get('_verification_task', {}),
        "process_audit_version": review.get('_process_audit_version', ''),
        "independent_verification": review.get("_independent_verification", {}),
        "visual_premise": visual_premise,
        "visual_evidence_ids": visual_ids,
        "caption_premise": caption_premise,
        "caption_evidence_ids": caption_ids,
        "caption_contract_source": f"claim_contract:{contract.get('schema_version', 'unknown')}",
        "bridge_type": family,
        "declared_bridge_type": declared_family,
        "bridge_family_source": "typed_claim_structure",
        "bridge_family_overridden": bool(
            declared_family and declared_family != family
        ),
        "bridge_statement": bridge_statement,
        "same_entity_check": None,
        "same_scope_check": None,
        "proposed_relation": relation,
        "verification_status": level,
        "provenance_roots": visual_ids + caption_ids,
        "counter_interpretation": counter,
        "counter_interpretation_strength": review.get(
            "counter_interpretation_strength", 1.0 if counter else 0.0
        ),
        "support_case": _clean(review.get("support_case")),
        "support_evidence_ids": list(review.get("support_evidence_ids", []) or []),
        "support_strength": review.get("support_strength"),
        "conflict_case": _clean(review.get("conflict_case")),
        "conflict_evidence_ids": list(review.get("conflict_evidence_ids", []) or []),
        "conflict_strength": review.get("conflict_strength"),
        "symmetric_cases_present": bool(review.get("symmetric_cases_present", False)),
        "confidence": confidence,
        "declared_admissibility": str(
            review.get("admissibility") or "INSUFFICIENT"
        ).strip().upper(),
        "deficiencies": deficiencies,
    }
