"""Compact, symmetric evidence packets for Arbiter and debate review."""

from copy import deepcopy


SUPPORT_TYPES = {"direct_support", "verified_support"}
CONFLICT_TYPES = {"direct_conflict", "verified_conflict"}


def _clean(value):
    return " ".join(str(value or "").split())


def _items(value):
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value if _clean(item)]
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def build_decision_packet(language_output, comparison):
    """Return a JSON-safe packet with equal support and conflict branches."""
    language_output = language_output or {}
    comparison = comparison or {}
    contract = language_output.get("claim_contract", {}) or {}
    relation = comparison.get("claim_relation", {}) or {}
    support = [
        {"id": None, "text": text, "decision_grade": True}
        for text in _items(comparison.get("supporting_evidence"))
    ]
    conflict = [
        {"id": None, "text": text, "decision_grade": True}
        for text in _items(comparison.get("contradicting_evidence"))
    ]

    def append_unique(target, item):
        key = (item.get("id"), _clean(item.get("text")).casefold())
        if any(
            (current.get("id"), _clean(current.get("text")).casefold()) == key
            for current in target
        ):
            return
        target.append(item)

    for item in comparison.get("grounded_evidence_catalog", []) or []:
        if not isinstance(item, dict):
            continue
        evidence = {
            "id": item.get("id"),
            "text": _clean(item.get("text")),
            "decision_grade": bool(item.get("decision_grade", False)),
            "source": item.get("source"),
            "verification_method": item.get("verification_method"),
        }
        kind = str(item.get("type", "")).lower()
        if kind in SUPPORT_TYPES:
            append_unique(support, evidence)
        elif kind in CONFLICT_TYPES:
            append_unique(conflict, evidence)

    return {
        "selected_claim": _clean(
            contract.get("selected_proposition")
            or language_output.get("caption_proposition")
        ),
        "literal_claim": _clean(
            contract.get("literal_proposition")
            or language_output.get("literal_proposition")
        ),
        "pragmatic_claim": _clean(
            contract.get("pragmatic_proposition")
            or language_output.get("pragmatic_proposition")
        ),
        "interpretation_status": contract.get("interpretation_status", "legacy"),
        "claim_contract_valid": bool(
            contract.get("safe_for_directional_reasoning", False)
        ),
        "support_hypothesis": deepcopy(
            comparison.get("support_hypothesis")
            or relation.get("support_hypothesis")
            or {}
        ),
        "conflict_hypothesis": deepcopy(
            comparison.get("conflict_hypothesis")
            or relation.get("conflict_hypothesis")
            or {}
        ),
        "support_evidence": support,
        "conflict_evidence": conflict,
        "grounded_anchors": _items(comparison.get("grounded_anchor_evidence")),
        "missing_evidence": _items(comparison.get("missing_evidence")),
        "evidence_status": comparison.get("required_evidence_status"),
    }


def format_decision_packet(packet):
    packet = packet or {}

    def hypothesis(name):
        item = packet.get(name, {}) or {}
        return (
            f"entity cues={item.get('entity_cues', [])}; "
            f"state={item.get('state', 'unresolved')}; "
            f"state cues={item.get('state_cues', [])}"
        )

    def evidence_lines(name):
        values = packet.get(name, []) or []
        if not values:
            return "- NONE"
        lines = []
        for item in values[:8]:
            evidence_id = item.get("id") or "NO_ID"
            grade = "decision-grade" if item.get("decision_grade") else "anchor-only"
            lines.append(f"- [{evidence_id}] ({grade}) {item.get('text', '')}")
        return "\n".join(lines)

    return "\n".join([
        "STRUCTURED DECISION PACKET",
        f"Selected claim: {packet.get('selected_claim') or 'unresolved'}",
        f"Interpretation status: {packet.get('interpretation_status')}",
        f"Claim contract valid: {packet.get('claim_contract_valid')}",
        f"SUPPORT hypothesis: {hypothesis('support_hypothesis')}",
        "SUPPORT evidence:",
        evidence_lines("support_evidence"),
        f"CONFLICT hypothesis: {hypothesis('conflict_hypothesis')}",
        "CONFLICT evidence:",
        evidence_lines("conflict_evidence"),
        f"Evidence status: {packet.get('evidence_status')}",
        "Rule: NONE means unavailable evidence, not evidence for the other branch.",
    ])
