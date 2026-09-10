"""Caption-span diagnostics, not a general semantic-equivalence oracle."""
import re


def _normal(value):
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def audit_claim_frame(caption, fields):
    source = str(caption or "")
    spans = {}
    for role in ("claim_subject", "claim_predicate", "claim_object", "claim_source", "claim_target",
                 "negation", "quantities", "claim_modifiers", "comparison_direction", "time_or_panel_scope"):
        values = fields.get(role)
        values = values if isinstance(values, list) else [values]
        matches = {}
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            for match in re.finditer(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", source, re.I):
                matches[(match.start(), match.end())] = {
                    "start": match.start(), "end": match.end(), "text": source[match.start():match.end()]}
        spans[role] = list(matches.values())
    subject, predicate, obj = (_normal(fields.get(key)) for key in (
        "claim_subject", "claim_predicate", "claim_object"))
    normalized = _normal(source)
    # This only rejects a complete exact inverse SVO frame. Partial token
    # order, passives, ellipsis and figurative mappings are not certified by
    # this check and require independent semantic review.
    swapped = bool(subject and predicate and obj and subject != obj
                   and normalized == f"{obj} {predicate} {subject}")
    exact = bool(subject and predicate and obj
                 and normalized == f"{subject} {predicate} {obj}")
    return {"source_spans": spans, "role_swap_detected": swapped,
            "status": "ROLE_REVERSAL" if swapped else "EXACT_SVO" if exact else "SEMANTIC_REVIEW_REQUIRED",
            "semantic_equivalence_verified": False}
