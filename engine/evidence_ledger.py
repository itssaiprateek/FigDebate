"""Deterministic evidence provenance for FigDebate decisions and reviews."""

from copy import deepcopy

from engine.relation_schema import nominate_visual_relations


RELATION_FOR_LABEL = {
    "ENTAILS": "SUPPORT",
    "CONTRADICTS": "CONFLICT",
}


def _clean(value):
    return " ".join(str(value or "").split())


def _append(
    entries,
    prefix,
    source,
    kind,
    text,
    relation="NEUTRAL",
    grounded=True,
    decision_grade=False,
    verification_method=None,
):
    text = _clean(text)
    if not text:
        return
    if any(
        item["source"] == source
        and item["type"] == kind
        and item["text"].casefold() == text.casefold()
        for item in entries
    ):
        return
    number = 1 + sum(item["id"].startswith(prefix) for item in entries)
    entries.append({
        "id": f"{prefix}{number:03d}",
        "source": source,
        "type": kind,
        "text": text,
        "relation": relation,
        "grounded": bool(grounded),
        "decision_grade": bool(decision_grade),
        "verification_method": verification_method,
    })


def build_evidence_ledger(visual_output, language_output, comparison):
    """Create an immutable, JSON-serializable ledger from existing outputs."""
    entries = []
    for text in visual_output.get("visual_facts", []) or []:
        _append(entries, "VF", "agent1", "visual_fact", text)
    for text in visual_output.get("visual_relations", []) or []:
        _append(entries, "VR", "agent1", "visual_relation", text)
    for text in visual_output.get("visible_text", []) or []:
        _append(entries, "VT", "agent1", "visible_text", text)
    for binding in visual_output.get("entity_state_bindings", []) or []:
        if not isinstance(binding, dict) or not binding.get("complete", False):
            continue
        _append(
            entries,
            "VB",
            "agent1",
            "entity_state_binding",
            binding.get("source_text") or (
                f"{binding.get('entity', '')} {binding.get('state', '')}"
            ),
            relation="NEUTRAL",
            grounded=bool(binding.get("grounded", False)),
            decision_grade=False,
            verification_method=binding.get("method"),
        )
    symbolic_tone = str(visual_output.get("symbolic_tone", "")).strip()
    if symbolic_tone.lower() not in {"", "none", "unclear", "unspecified"}:
        _append(
            entries, "VS", "agent1", "symbolic_anchor", symbolic_tone,
            relation="ANCHOR", grounded=True, decision_grade=False,
        )
    for text in visual_output.get("possible_visual_metaphors", []) or []:
        _append(
            entries, "VH", "agent1", "visual_hypothesis", text,
            relation="NEUTRAL", grounded=False, decision_grade=False,
        )
    for text in visual_output.get("uncertain_observations", []) or []:
        _append(
            entries, "VU", "agent1", "uncertain_observation", text,
            grounded=False,
        )

    _append(
        entries,
        "LC",
        "agent2",
        "caption_proposition",
        language_output.get("caption_proposition"),
        grounded=False,
    )
    _append(
        entries,
        "LM",
        "agent2",
        "intended_meaning",
        language_output.get("intended_meaning"),
        grounded=False,
    )

    for text in comparison.get("supporting_evidence", []) or []:
        _append(
            entries,
            "CS",
            "comparator",
            "direct_support",
            text,
            "SUPPORT",
            decision_grade=True,
            verification_method="deterministic_explicit_relation",
        )
    for text in comparison.get("contradicting_evidence", []) or []:
        _append(
            entries,
            "CC",
            "comparator",
            "direct_conflict",
            text,
            "CONFLICT",
            decision_grade=True,
            verification_method="deterministic_explicit_relation",
        )
    for text in comparison.get("grounded_anchor_evidence", []) or []:
        _append(entries, "CA", "comparator", "grounded_anchor", text, "ANCHOR")
    for text in comparison.get("missing_evidence", []) or []:
        _append(
            entries, "CM", "comparator", "missing_evidence", text,
            relation="MISSING", grounded=False,
        )
    return entries


def evidence_ids(ledger, relation=None, grounded_only=True):
    return [
        item["id"]
        for item in (ledger or [])
        if (relation is None or item.get("relation") == relation)
        and (not grounded_only or item.get("grounded", False))
    ]


def audit_decision(decision, ledger):
    """Separate valid source attribution from independently verified direction."""
    label = decision.get("label")
    relation = RELATION_FOR_LABEL.get(label)
    by_id = {item.get("id"): item for item in (ledger or [])}
    claimed = list(dict.fromkeys(
        str(item).strip().upper()
        for item in decision.get("_model_cited_evidence_ids", []) or []
        if str(item).strip()
    ))
    source_cited = [
        item_id for item_id in claimed
        if item_id in by_id
        and by_id[item_id].get("grounded", False)
        and by_id[item_id].get("source") in {
            "agent1",
            "comparator",
            "targeted_region_verifier",
            "debate_visual_reinspection",
        }
    ]
    cited = [
        item_id for item_id in source_cited
        if by_id[item_id].get("relation") == relation
        and (
            by_id[item_id].get("decision_grade", False)
            or by_id[item_id].get("verification", {}).get(
                "decision_grade", False
            )
        )
    ] if relation else []
    # A pinned targeted region verifier is already an independent directional
    # check and may be attached after the Arbiter generated its assessment.
    if decision.get("decision_method") == "targeted_region_verifier" and relation:
        targeted_ids = [
            item_id for item_id in evidence_ids(ledger, relation)
            if by_id[item_id].get("source") == "targeted_region_verifier"
            and by_id[item_id].get("decision_grade", False)
        ]
        source_cited.extend(
            item_id for item_id in targeted_ids if item_id not in source_cited
        )
        cited.extend(item_id for item_id in targeted_ids if item_id not in cited)
    invalid_cited = [item_id for item_id in claimed if item_id not in source_cited]
    anchors = evidence_ids(ledger, "ANCHOR")
    if cited:
        status = "DIRECT_GROUNDED_EVIDENCE"
    elif source_cited:
        status = "GROUNDED_SOURCE_CITED"
    elif anchors:
        status = "GROUNDED_ANCHORS_ONLY"
    else:
        status = "NO_DECISION_GRADE_EVIDENCE"
    return {
        "label": label,
        "required_relation": relation,
        "valid": bool(cited),
        "source_valid": bool(source_cited),
        "status": status,
        "cited_evidence_ids": cited,
        "source_cited_evidence_ids": source_cited,
        "invalid_cited_evidence_ids": invalid_cited,
        "grounded_anchor_ids": anchors,
    }


def attach_evidence_audit(decision, ledger):
    output = dict(decision)
    output["_evidence_ledger"] = deepcopy(ledger or [])
    output["_evidence_audit"] = audit_decision(output, ledger)
    output["_cited_evidence_ids"] = output["_evidence_audit"][
        "cited_evidence_ids"
    ]
    output["_source_cited_evidence_ids"] = output["_evidence_audit"][
        "source_cited_evidence_ids"
    ]
    return output


def add_targeted_verifier_evidence(ledger, critique, revised_decision):
    """Add region evidence with an explicit verification grade."""
    output = deepcopy(ledger or [])
    if revised_decision.get("decision_method") != "targeted_region_verifier":
        return output
    verification = revised_decision.get("_targeted_region_verification", {}) or {}
    if verification.get("decision_grade", False):
        relation = verification.get("evidence_relation")
        if relation not in {"SUPPORT", "CONFLICT"}:
            return output
        reason = critique.get("reason", "") if isinstance(critique, dict) else ""
        _append(
            output,
            "TV",
            "targeted_region_verifier",
            "verified_region_relation",
            reason,
            relation=relation,
            grounded=True,
            decision_grade=True,
            verification_method=verification.get(
                "method", "deterministic_structured_region_binding"
            ),
        )
        return output
    scores = revised_decision.get("_binary_resolution_scores", {}) or {}
    if not scores.get("nli_model"):
        return output
    label = revised_decision.get("label")
    relation = RELATION_FOR_LABEL.get(label)
    if not relation:
        return output
    reason = critique.get("reason", "") if isinstance(critique, dict) else ""
    _append(
        output,
        "TV",
        "targeted_region_verifier",
        "nli_region_relation_candidate",
        reason,
        relation=relation,
        grounded=True,
        decision_grade=False,
        verification_method="generic_text_nli_diagnostic",
    )
    return output


def add_visual_reinspection_evidence(ledger, critique, comparison):
    """Promote only strict entity-bound cues from an independent reinspection."""
    output = deepcopy(ledger or [])
    if not isinstance(critique, dict) or not critique.get(
        "specific_evidence", False
    ):
        return output
    recommendation = str(critique.get("recommendation", "")).upper()
    expected_relation = {
        "ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"
    }.get(recommendation)
    if not expected_relation:
        return output
    if str(critique.get("claim_relation", "")).upper() != expected_relation:
        return output
    relation = (comparison or {}).get("claim_relation", {}) or {}
    observed_entity = str(critique.get("observed_entity", "")).strip()
    observed_state = str(critique.get("observed_state", "")).strip()
    image_region = str(critique.get("image_region", "")).strip()
    if not observed_entity or not observed_state:
        return output
    structured_observation = (
        f"{observed_entity} {observed_state}. "
        f"Region: {image_region}. {critique.get('reason', '')}"
    )
    nominations = nominate_visual_relations(
        {"visual_facts": [structured_observation]}, relation
    )
    strict = [
        item for item in nominations
        if item.get("proposed_relation") == expected_relation
        and len(set(item.get("matched_cues", []))) >= 1
        and item.get("matched_entities")
    ]
    if len(strict) != 1:
        return output
    item = strict[0]
    _append(
        output,
        "DV",
        "debate_visual_reinspection",
        "verified_reinspection_relation",
        structured_observation,
        relation=expected_relation,
        grounded=True,
        decision_grade=True,
        verification_method=(
            "structured_visual_reinspection_entity_bound_state"
        ),
    )
    return output
