"""Deterministic evidence provenance for FigDebate decisions and reviews."""

from copy import deepcopy

from engine.relation_schema import nominate_visual_relations


RELATION_FOR_LABEL = {
    "ENTAILS": "SUPPORT",
    "CONTRADICTS": "CONFLICT",
}
EVIDENCE_LIFECYCLE_STATUSES = {
    "ACTIVE", "DISPUTED", "SUPERSEDED", "RECONFIRMED", "REJECTED_FORMAT",
}
EVIDENCE_LEVELS = {
    "OBSERVATION", "BINDING", "RELATION_CANDIDATE", "VERIFIED_RELATION",
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
    lifecycle_status="ACTIVE",
    source_generation="initial",
    question_id=None,
    supersedes_ids=None,
    evidence_level=None,
    derived_from_ids=None,
    reliability=None,
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
    lifecycle_status = str(lifecycle_status or "ACTIVE").upper()
    if lifecycle_status not in EVIDENCE_LIFECYCLE_STATUSES:
        raise ValueError(f"Unknown evidence lifecycle status: {lifecycle_status}")
    if evidence_level is None:
        evidence_level = (
            "VERIFIED_RELATION" if decision_grade
            else "RELATION_CANDIDATE" if relation in {"SUPPORT", "CONFLICT"}
            else "OBSERVATION"
        )
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"Unknown evidence level: {evidence_level}")
    if decision_grade and evidence_level != "VERIFIED_RELATION":
        raise ValueError("Decision-grade evidence must be a VERIFIED_RELATION.")
    entries.append({
        "id": f"{prefix}{number:03d}",
        "source": source,
        "type": kind,
        "text": text,
        "relation": relation,
        "grounded": bool(grounded),
        "decision_grade": bool(decision_grade),
        "verification_method": verification_method,
        "lifecycle_status": lifecycle_status,
        "source_generation": source_generation,
        "question_id": question_id,
        "supersedes_ids": list(supersedes_ids or []),
        "evidence_level": evidence_level,
        "derived_from_ids": list(derived_from_ids or []),
        "reliability": reliability,
    })


def is_active_evidence(item):
    return str(item.get("lifecycle_status", "ACTIVE")).upper() in {
        "ACTIVE", "RECONFIRMED"
    }


VERIFICATION_RELIABILITY = {
    "deterministic_structured_region_binding": 1.0,
    "deterministic_numeric_or_geometric_relation": 1.0,
    "independent_multimodal_corroboration": 0.90,
    "cross_agent_structured_relation": 0.78,
    "tribunal_semantic_corroboration": 0.74,
    "tribunal_normative_corroboration": 0.68,
    "structured_visual_reinspection_entity_bound_state": 0.70,
    "generic_text_nli_diagnostic": 0.25,
    "structured_lexical_nomination": 0.20,
}


def evidence_reliability(item):
    """Score evidence by verification independence, never by item count."""
    explicit = item.get("reliability")
    if isinstance(explicit, (int, float)):
        return max(0.0, min(float(explicit), 1.0))
    method = item.get("verification_method") or (
        item.get("verification", {}) or {}
    ).get("method")
    return VERIFICATION_RELIABILITY.get(
        method,
        0.60 if item.get("decision_grade", False) else 0.10,
    )


def evidence_provenance_roots(ledger, item_or_id):
    """Return transitive source roots for independence-aware aggregation."""
    by_id = {item.get("id"): item for item in (ledger or []) if item.get("id")}
    item = by_id.get(item_or_id, {}) if isinstance(item_or_id, str) else (
        item_or_id or {}
    )

    def visit(current, trail):
        current_id = current.get("id")
        if current_id in trail:
            return {current_id} if current_id else set()
        parents = [
            parent for parent in (current.get("derived_from_ids", []) or [])
            if parent in by_id
        ]
        if not parents:
            return {current_id} if current_id else set()
        roots = set()
        for parent in parents:
            roots.update(visit(by_id[parent], trail | {current_id}))
        return roots

    return sorted(visit(item, set()))


def promote_verified_relation(
    ledger, *, source, text, relation, method, derived_from_ids=None,
    reliability=None, question_id=None,
):
    """Append one independently verified relation with full provenance."""
    if relation not in {"SUPPORT", "CONFLICT"}:
        raise ValueError("Verified relation must be SUPPORT or CONFLICT.")
    if method not in {
        "deterministic_structured_region_binding",
        "deterministic_numeric_or_geometric_relation",
        "independent_multimodal_corroboration",
        "cross_agent_structured_relation",
        "tribunal_semantic_corroboration",
        "tribunal_normative_corroboration",
    }:
        raise ValueError("Unapproved verification method cannot promote evidence.")
    output = deepcopy(ledger or [])
    _append(
        output,
        (
            "TV" if method.startswith("deterministic")
            else "AV" if method == "cross_agent_structured_relation"
            else "IV"
        ),
        source,
        "verified_relation",
        text,
        relation=relation,
        grounded=True,
        decision_grade=True,
        verification_method=method,
        evidence_level="VERIFIED_RELATION",
        derived_from_ids=derived_from_ids,
        reliability=(
            reliability
            if reliability is not None
            else VERIFICATION_RELIABILITY[method]
        ),
        question_id=question_id,
        source_generation="tribunal_verification",
    )
    return output


def add_cross_agent_verified_relation(
    ledger, visual_critique, claim_critique, claim_contract
):
    """Promote only agreement between a visual witness and a safe claim frame.

    Agent 1 supplies the current-image observation and a relation candidate.
    Agent 2 independently supplies and validates the mutually opposing caption
    conditions.  Neither model may promote its own output alone.
    """
    output = deepcopy(ledger or [])
    visual = visual_critique or {}
    claim = claim_critique or {}
    contract = claim_contract or {}
    relation = str(visual.get("claim_relation", "")).upper()
    witness = visual.get("witness_contract", {}) or {}
    if relation not in {"SUPPORT", "CONFLICT"}:
        return output, {"promoted": False, "reason": "visual_relation_unresolved"}
    if visual.get("response_status") != "VALID_DIRECTIONAL_ANSWER":
        return output, {"promoted": False, "reason": "visual_answer_not_directional"}
    if not visual.get("specific_evidence", False):
        return output, {"promoted": False, "reason": "visual_evidence_not_specific"}
    if (
        witness.get("answer_status") != "OBSERVED"
        or not witness.get("direction_assigned", False)
        or str(witness.get("relation_candidate", "")).upper() != relation
    ):
        return output, {
            "promoted": False,
            "reason": "visual_witness_contract_inconsistent",
        }
    if not claim.get("_format_valid", False):
        return output, {"promoted": False, "reason": "claim_audit_format_invalid"}
    if not claim.get("requirements_valid", False):
        return output, {"promoted": False, "reason": "claim_requirements_invalid"}
    if str(claim.get("stance", "")).upper() != "ENDORSE":
        return output, {"promoted": False, "reason": "claim_frame_not_endorsed"}
    if not contract.get("safe_for_automatic_directional_reasoning", False):
        return output, {"promoted": False, "reason": "claim_not_automatic_direction_safe"}

    question_id = visual.get("question_id")
    observation = " ".join(str(visual.get("observed_state", "")).split())
    witness_observation = " ".join(
        str(witness.get("observation", "")).split()
    )
    if not observation or observation != witness_observation:
        return output, {
            "promoted": False,
            "reason": "visual_witness_observation_mismatch",
        }
    witness_ids = [
        item.get("id") for item in output
        if item.get("source") == "debate_visual_witness"
        and item.get("grounded", False)
        and " ".join(str(item.get("text", "")).split()) == observation
        and (
            not question_id or item.get("question_id") == question_id
        )
    ]
    if not witness_ids:
        return output, {"promoted": False, "reason": "visual_witness_not_recorded"}

    requirement = (
        claim.get("support_requirement")
        if relation == "SUPPORT" else claim.get("conflict_requirement")
    )
    text = (
        f"Observed: {observation}. Caption condition: "
        f"{' '.join(str(requirement or '').split())}."
    )
    promoted = promote_verified_relation(
        output,
        source="cross_agent_relation_verifier",
        text=text,
        relation=relation,
        method="cross_agent_structured_relation",
        derived_from_ids=witness_ids,
        reliability=VERIFICATION_RELIABILITY["cross_agent_structured_relation"],
        question_id=question_id,
    )
    promoted[-1]["validation_sources"] = [
        "agent1_visual_witness",
        "agent2_claim_audit",
        "typed_claim_contract",
    ]
    promoted[-1]["claim_audit"] = {
        "stance": str(claim.get("stance", "")).upper(),
        "format_valid": bool(claim.get("_format_valid", False)),
        "requirements_valid": bool(claim.get("requirements_valid", False)),
    }
    promoted[-1]["claim_contract_schema_version"] = contract.get(
        "schema_version"
    )
    return promoted, {
        "promoted": True,
        "reason": "independent_visual_and_claim_contract_agree",
        "evidence_id": promoted[-1]["id"],
        "relation": relation,
    }


def add_tribunal_corroborated_relation(
    ledger, review, claim_contract, visual_critique, claim_critique,
):
    """Promote a relation only from three separately auditable premises.

    Agent 1 supplies a current-round visual witness, Agent 2 validates the
    immutable caption condition, and the independent tribunal judge assigns
    the relation.  The judge's own prose is never promoted as visual evidence.
    """
    output = deepcopy(ledger or [])
    review = review or {}
    contract = claim_contract or {}
    visual = visual_critique or {}
    claim = claim_critique or {}
    relation = str(review.get("relation", "")).upper()
    if review.get("status") != "RESOLVE" or relation not in {
        "SUPPORT", "CONFLICT"
    }:
        return output, {"promoted": False, "reason": "tribunal_relation_unresolved"}
    if not review.get("_format_valid", False):
        return output, {"promoted": False, "reason": "tribunal_contract_invalid"}
    if not contract.get(
        "safe_for_tribunal_reasoning",
        contract.get("safe_for_directional_reasoning", False),
    ):
        return output, {"promoted": False, "reason": "claim_contract_not_safe"}
    if visual.get("response_status") and visual.get("response_status") not in {
        "VALID_OBSERVATION", "VALID_DIRECTIONAL_ANSWER"
    }:
        return output, {
            "promoted": False, "reason": "visual_witness_response_invalid"
        }
    if not (
        claim.get("_format_valid", False)
        and claim.get("requirements_valid", False)
        and str(claim.get("stance", "")).upper() == "ENDORSE"
    ):
        return output, {"promoted": False, "reason": "claim_audit_not_endorsed"}
    try:
        confidence = float(review.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    normative = bool(contract.get("requires_normative_reasoning", False))
    minimum_confidence = 0.82 if normative else 0.78
    if confidence < minimum_confidence:
        return output, {
            "promoted": False, "reason": "tribunal_corroboration_below_threshold"
        }

    by_id = {item.get("id"): item for item in output}
    cited = set(review.get("_valid_evidence_ids", []) or [])
    question_id = visual.get("question_id")
    if not question_id:
        return output, {
            "promoted": False, "reason": "current_round_question_id_missing"
        }
    witness_ids = [
        item_id for item_id in cited
        if by_id.get(item_id, {}).get("source") == "debate_visual_witness"
        and by_id.get(item_id, {}).get("grounded", False)
        and is_active_evidence(by_id.get(item_id, {}))
        and (
            not question_id
            or by_id.get(item_id, {}).get("question_id") == question_id
        )
    ]
    if not witness_ids:
        return output, {
            "promoted": False, "reason": "current_round_visual_witness_not_cited"
        }
    visual_relation = str(visual.get("claim_relation", "")).upper()
    if visual_relation in {"SUPPORT", "CONFLICT"} and visual_relation != relation:
        return output, {
            "promoted": False, "reason": "visual_and_tribunal_relation_disagree"
        }

    caption_ids = [
        item.get("id") for item in output
        if item.get("source") == "agent2"
        and item.get("type") == "caption_proposition"
        and item.get("id")
    ][:1]
    if not caption_ids:
        return output, {
            "promoted": False, "reason": "caption_proposition_not_recorded"
        }
    method = (
        "tribunal_normative_corroboration" if normative
        else "tribunal_semantic_corroboration"
    )
    observation = " ".join(
        " ".join(str(by_id[item_id].get("text", "")).split())
        for item_id in witness_ids
    )
    promoted = promote_verified_relation(
        output,
        source="tribunal_relation_verifier",
        text=(
            f"Observed: {observation}. Corroborated relation to preserved "
            f"caption proposition: {relation}."
        ),
        relation=relation,
        method=method,
        derived_from_ids=witness_ids + caption_ids,
        reliability=VERIFICATION_RELIABILITY[method],
        question_id=question_id,
    )
    promoted[-1]["validation_sources"] = [
        "agent1_current_round_witness",
        "agent2_preserved_claim_audit",
        "independent_tribunal_relation",
    ]
    return promoted, {
        "promoted": True,
        "reason": "three_source_relation_corroboration",
        "evidence_id": promoted[-1]["id"],
        "relation": relation,
        "method": method,
    }


def evidence_lifecycle_summary(ledger):
    summary = {}
    for item in ledger or []:
        status = str(item.get("lifecycle_status", "ACTIVE")).upper()
        summary[status] = summary.get(status, 0) + 1
    return summary


def build_evidence_ledger(visual_output, language_output, comparison):
    """Create an immutable, JSON-serializable ledger from existing outputs."""
    entries = []
    for text in visual_output.get("visual_facts", []) or []:
        _append(entries, "VF", "agent1", "visual_fact", text)
    for text in visual_output.get("visual_relations", []) or []:
        _append(entries, "VR", "agent1", "visual_relation", text)
    for text in visual_output.get("visible_text", []) or []:
        _append(entries, "VT", "agent1", "visible_text", text)
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

    for record in comparison.get("structured_observations", []) or []:
        if not record.get("text"):
            continue
        _append(
            entries,
            "SB",
            "structured_observation_builder",
            str(record.get("record_type", "binding")).casefold(),
            record.get("text"),
            relation="ANCHOR",
            grounded=True,
            decision_grade=False,
            verification_method="deterministic_observation_typing",
            evidence_level="BINDING",
            derived_from_ids=[],
            reliability=0.55 if record.get("binding_complete") else 0.35,
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

    for text in comparison.get("relation_support_candidates", []) or []:
        _append(
            entries,
            "CS",
            "comparator",
            "direct_support",
            text,
            "SUPPORT",
            decision_grade=False,
            verification_method="structured_lexical_nomination",
            evidence_level="RELATION_CANDIDATE",
            reliability=0.20,
        )
    for text in comparison.get("relation_conflict_candidates", []) or []:
        _append(
            entries,
            "CC",
            "comparator",
            "direct_conflict",
            text,
            "CONFLICT",
            decision_grade=False,
            verification_method="structured_lexical_nomination",
            evidence_level="RELATION_CANDIDATE",
            reliability=0.20,
        )
    for text in comparison.get("grounded_anchor_evidence", []) or []:
        _append(entries, "CA", "comparator", "grounded_anchor", text, "ANCHOR")
    for text in comparison.get("missing_evidence", []) or []:
        _append(
            entries, "CM", "comparator", "missing_evidence", text,
            relation="MISSING", grounded=False,
        )
    generation = (
        "targeted_recovery"
        if visual_output.get("_targeted_recovery_attempted", False)
        else "initial"
    )
    for item in entries:
        if item.get("source") in {"agent1", "comparator"}:
            item["source_generation"] = generation
    return entries


def evidence_ids(ledger, relation=None, grounded_only=True):
    return [
        item["id"]
        for item in (ledger or [])
        if (relation is None or item.get("relation") == relation)
        and (not grounded_only or item.get("grounded", False))
        and is_active_evidence(item)
    ]


def audit_decision(decision, ledger):
    """Separate valid source attribution from independently verified direction."""
    label = decision.get("label")
    relation = RELATION_FOR_LABEL.get(label)
    by_id = {
        item.get("id"): item for item in (ledger or [])
        if is_active_evidence(item)
    }
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
            "debate_visual_witness",
            "tribunal_independent_verifier",
            "cross_agent_relation_verifier",
            "tribunal_relation_verifier",
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
    """Record a strict reinspection as a candidate pending corroboration.

    The observation and relation are produced by the same visual model, so
    they are not independent verification and cannot be decision grade.
    """
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
    same_direction_exists = any(
        is_active_evidence(existing)
        and existing.get("relation") == expected_relation
        and existing.get("grounded", False)
        and (
            existing.get("decision_grade", False)
            or existing.get("verification", {}).get("decision_grade", False)
        )
        for existing in output
    )
    opposite = "CONFLICT" if expected_relation == "SUPPORT" else "SUPPORT"
    for existing in output:
        if (
            existing.get("source") == "debate_visual_reinspection"
            and existing.get("relation") == opposite
            and is_active_evidence(existing)
        ):
            existing["lifecycle_status"] = "DISPUTED"
    _append(
        output,
        "DV",
        "debate_visual_reinspection",
        "verified_reinspection_relation",
        structured_observation,
        relation=expected_relation,
        grounded=True,
        decision_grade=False,
        verification_method="single_model_visual_relation_candidate",
        evidence_level="RELATION_CANDIDATE",
        reliability=0.45,
        source_generation="debate_reinspection",
        question_id=critique.get("question_id"),
        lifecycle_status=("RECONFIRMED" if same_direction_exists else "ACTIVE"),
    )
    return output


def add_visual_witness_evidence(ledger, critique):
    """Record a validated answer without assigning a semantic direction."""
    output = deepcopy(ledger or [])
    witness = (critique or {}).get("witness_contract", {}) or {}
    observation = _clean(
        witness.get("observation") or (critique or {}).get("observed_state")
    )
    if (
        not observation
        or witness.get("answer_status") != "OBSERVED"
        or not (critique or {}).get("_format_valid", False)
    ):
        return output
    _append(
        output,
        "DW",
        "debate_visual_witness",
        "entity_bound_observation",
        observation,
        relation="NEUTRAL",
        grounded=True,
        decision_grade=False,
        verification_method="atomic_visual_witness_answer",
        evidence_level="BINDING",
        reliability=0.55,
        source_generation="tribunal_witness",
        question_id=witness.get("question_id") or critique.get("question_id"),
    )
    return output
