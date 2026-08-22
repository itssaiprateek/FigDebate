"""Conservative deterministic verification for region-bound text relations."""

import re


STOPWORDS = {
    "about", "after", "also", "and", "are", "been", "being", "for",
    "from", "has", "have", "into", "its", "object", "outcome", "same",
    "that", "the", "their", "there", "this", "those", "was", "were",
    "with",
}
TOKEN_ALIASES = {
    "disliked": "dislike", "dislikes": "dislike", "hated": "dislike",
    "hates": "dislike", "hate": "dislike", "liked": "love",
    "likes": "love", "loved": "love", "loves": "love",
    "items": "product", "products": "product",
}
PREFERENCE_CUES = {
    "dislike": {"dislike"},
    "love": {"love", "favorite", "favourite"},
}
DURATION_CUES = {
    "short": {"disappear", "disappears", "fast", "gone", "quick", "quickly", "week", "weekly"},
    "long": {"forever", "last", "lasts", "long", "month", "months"},
}


def _tokens(value):
    return {
        TOKEN_ALIASES.get(token, token)
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


def _best_overlap(observed, clauses):
    overlaps = [len(observed & clause) for clause in clauses]
    return max(overlaps, default=0)


def _single_pole(tokens, cue_map):
    matches = [
        pole for pole, cues in cue_map.items()
        if tokens.intersection(cues)
    ]
    return matches[0] if len(matches) == 1 else None


def _preference_duration_map(text):
    mapping = {}
    clauses = re.split(
        r"\s*(?:,|;|\||\bwhile\b|\bwhereas\b|\bbut\b)\s*",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    for clause in clauses:
        tokens = _tokens(clause)
        preference = _single_pole(tokens, PREFERENCE_CUES)
        duration = _single_pole(tokens, DURATION_CUES)
        if preference and duration:
            mapping[preference] = duration
    return mapping


def _verify_preference_duration(region_pairs, relation):
    expected = _preference_duration_map(relation.get("claim_text", ""))
    if set(expected) != {"dislike", "love"}:
        return None

    observed = {}
    pair_results = []
    for pair in region_pairs:
        tokens = _tokens(
            f"{pair.get('object_text', '')} {pair.get('outcome_text', '')}"
        )
        preference = _single_pole(tokens, PREFERENCE_CUES)
        duration = _single_pole(tokens, DURATION_CUES)
        pair_results.append({
            "side": str(pair.get("side", "region")),
            "object_text": str(pair.get("object_text", "")),
            "outcome_text": str(pair.get("outcome_text", "")),
            "preference": preference,
            "duration": duration,
        })
        if preference and duration:
            observed[preference] = duration

    if set(observed) != set(expected):
        return None
    matches = sum(observed[key] == expected[key] for key in expected)
    inversions = sum(observed[key] != expected[key] for key in expected)
    if matches == len(expected):
        evidence_relation = "SUPPORT"
    elif inversions == len(expected):
        evidence_relation = "CONFLICT"
    else:
        return None
    return {
        "resolved": True,
        "decision_grade": True,
        "evidence_relation": evidence_relation,
        "label": "ENTAILS" if evidence_relation == "SUPPORT" else "CONTRADICTS",
        "confidence": 0.9,
        "reason": "complete_preference_duration_region_binding",
        "pair_results": pair_results,
        "method": "deterministic_preference_duration_binding",
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
    if not contract.get("safe_for_directional_reasoning", False):
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

    preference_duration = _verify_preference_duration(region_pairs, relation)
    if preference_duration:
        return preference_duration

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
        observed = _tokens(f"{object_text} {outcome_text}")
        expected_score = _best_overlap(observed, expected)
        opposite_score = _best_overlap(observed, opposite)
        direction = "ABSTAIN"
        if max(expected_score, opposite_score) >= 2:
            if expected_score > opposite_score:
                direction = "SUPPORT"
            elif opposite_score > expected_score:
                direction = "CONFLICT"
        pair_results.append({
            "side": str(pair.get("side", "region")),
            "object_text": object_text,
            "outcome_text": outcome_text,
            "expected_overlap": expected_score,
            "opposite_overlap": opposite_score,
            "direction": direction,
        })

    informative = [
        item for item in pair_results if item["direction"] != "ABSTAIN"
    ]
    directions = {item["direction"] for item in informative}
    if not informative or len(directions) != 1:
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
