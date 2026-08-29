"""Conservative deterministic verification for region-bound text relations."""

import re


STOPWORDS = {
    "about", "after", "also", "and", "are", "been", "being", "for",
    "from", "has", "have", "into", "its", "object", "outcome", "same",
    "that", "the", "their", "there", "this", "those", "was", "were",
    "with",
}
def _tokens(value):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _clauses(value):
    parts = re.split(
        r"\s*(?:;|\||\bwhile\b|\bwhereas\b|\bbut\b|\band\b)\s*",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    return [_tokens(part) for part in parts if _tokens(part)]


def _bound_overlap(object_tokens, outcome_tokens, clauses):
    """Score the outcome only after uniquely binding the observed entity.

    Matching a state word in the wrong clause is not evidence.  When two
    clauses tie for the best entity match, the binding is ambiguous and the
    verifier abstains instead of selecting whichever outcome happens to fit.
    """
    scored = [
        {
            "entity_overlap": len(object_tokens & clause),
            "outcome_overlap": len(outcome_tokens & clause),
        }
        for clause in clauses
    ]
    best_entity = max(
        (item["entity_overlap"] for item in scored), default=0
    )
    bound = [
        item for item in scored
        if item["entity_overlap"] == best_entity and best_entity > 0
    ]
    if len(bound) != 1:
        return {
            "entity_overlap": best_entity,
            "outcome_overlap": 0,
            "binding_unambiguous": False,
        }
    return {
        **bound[0],
        "binding_unambiguous": True,
    }


def verify_region_pairs(region_pairs, claim_relation):
    """Resolve only explicit object-to-outcome bindings; otherwise abstain.

    This verifier deliberately avoids semantic model scores. It compares each
    observed region pair against the claim's expected and opposite states and
    requires a strict lexical margin. The result is decision-grade only when
    the immutable claim contract is valid and every informative pair points in
    the same direction.
    """
    relation = claim_relation or {}
    contract = relation.get("claim_contract", {}) or {}
    if not relation.get("resolved", False):
        return {
            "resolved": False,
            "decision_grade": False,
            "reason": "claim_relation_unresolved",
            "method": "deterministic_structured_region_binding",
        }
    if not contract.get("safe_for_automatic_directional_reasoning", False):
        return {
            "resolved": False,
            "decision_grade": False,
            "reason": "claim_contract_not_direction_safe",
            "method": "deterministic_structured_region_binding",
        }
    if not isinstance(region_pairs, list) or not region_pairs:
        return {
            "resolved": False,
            "decision_grade": False,
            "reason": "no_complete_region_pairs",
            "method": "deterministic_structured_region_binding",
        }

    expected = _clauses(relation.get("expected_visual_state"))
    opposite = _clauses(relation.get("opposite_visual_state"))
    if not expected or not opposite:
        return {
            "resolved": False,
            "decision_grade": False,
            "reason": "directional_state_pair_unresolved",
            "method": "deterministic_structured_region_binding",
        }

    pair_results = []
    for pair in region_pairs:
        object_text = str(pair.get("object_text", "")).strip()
        outcome_text = str(pair.get("outcome_text", "")).strip()
        if not object_text or not outcome_text:
            continue
        object_tokens = _tokens(object_text)
        outcome_tokens = _tokens(outcome_text)
        expected_binding = _bound_overlap(object_tokens, outcome_tokens, expected)
        opposite_binding = _bound_overlap(object_tokens, outcome_tokens, opposite)
        expected_score = expected_binding["outcome_overlap"]
        opposite_score = opposite_binding["outcome_overlap"]
        direction = "ABSTAIN"
        bindings_valid = bool(
            expected_binding["binding_unambiguous"]
            and opposite_binding["binding_unambiguous"]
        )
        # A region pair becomes directional only after an unambiguous entity
        # binding and a strict outcome advantage. Shared entity words never
        # contribute to the directional score.
        if bindings_valid and max(expected_score, opposite_score) >= 1:
            if expected_score >= opposite_score + 1:
                direction = "SUPPORT"
            elif opposite_score >= expected_score + 1:
                direction = "CONFLICT"
        pair_results.append({
            "side": str(pair.get("side", "region")),
            "object_text": object_text,
            "outcome_text": outcome_text,
            "expected_overlap": expected_score,
            "opposite_overlap": opposite_score,
            "expected_entity_overlap": expected_binding["entity_overlap"],
            "opposite_entity_overlap": opposite_binding["entity_overlap"],
            "binding_unambiguous": bindings_valid,
            "direction": direction,
        })

    informative = [
        item for item in pair_results if item["direction"] != "ABSTAIN"
    ]
    directions = {item["direction"] for item in informative}
    complete_pair_count = sum(
        bool(str(item.get("object_text", "")).strip())
        and bool(str(item.get("outcome_text", "")).strip())
        for item in region_pairs
    )
    if (
        not informative
        or len(directions) != 1
        or len(informative) != complete_pair_count
    ):
        return {
            "resolved": False,
            "decision_grade": False,
            "reason": (
                "mixed_region_directions" if len(directions) > 1
                else "insufficient_region_relation_overlap"
            ),
            "pair_results": pair_results,
            "method": "deterministic_structured_region_binding",
        }

    evidence_relation = informative[0]["direction"]
    return {
        "resolved": True,
        "decision_grade": True,
        "evidence_relation": evidence_relation,
        "label": "ENTAILS" if evidence_relation == "SUPPORT" else "CONTRADICTS",
        "confidence": min(0.95, 0.72 + 0.08 * len(informative)),
        "reason": "consistent_explicit_region_bindings",
        "pair_results": pair_results,
        "method": "deterministic_structured_region_binding",
    }
