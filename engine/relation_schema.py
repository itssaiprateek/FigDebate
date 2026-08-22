"""Conservative structured relations for cross-modal verification.

This module only nominates directions. A separate verifier must corroborate a
nomination before it becomes decision-grade evidence.
"""

import re
from copy import deepcopy

from engine.claim_contract import audit_claim_contract


RELATION_FAMILIES = {
    "trajectory": {
        "positive": ("grow", "growth", "rise", "rising", "increase", "recover", "improve", "progress", "upward"),
        "negative": ("decline", "drop", "fall", "falling", "decrease", "collapse", "crash", "downward"),
    },
    "pace": {
        "positive": ("fast", "quick", "quickly", "rapid", "rapidly", "running", "rushing"),
        "negative": ("slow", "slowly", "leisurely", "delayed", "delay", "waiting"),
    },
    "outcome": {
        "positive": ("success", "successful", "win", "winning", "achieve", "achievement", "works", "working"),
        "negative": ("failure", "failed", "fails", "lose", "losing", "broken", "error", "mistake"),
    },
    "sentiment": {
        "positive": (
            "happy", "smile", "smiling", "joy", "excited", "positive",
            "love", "loved", "healthy", "whole", "wholesome", "pure",
            "angelic", "intact",
        ),
        "negative": (
            "sad", "cry", "crying", "angry", "fear", "worried",
            "frustrated", "negative", "hate", "disliked", "rotten",
            "corrupt", "corrupted", "decayed", "damaged", "evil",
            "heartless",
        ),
    },
    "safety": {
        "positive": ("safe", "secure", "protected", "comfort", "comfortable"),
        "negative": ("unsafe", "danger", "dangerous", "threat", "weapon", "gun", "guns", "violence"),
    },
    "trust": {
        "positive": ("trust", "trusted", "trustworthy", "reliable", "dependable"),
        "negative": ("distrust", "untrustworthy", "unreliable", "misleading", "deceptive", "deceitful"),
    },
    "association": {
        "positive": ("together", "joined", "accompanied", "group", "alongside"),
        "negative": ("alone", "apart", "separate", "separated", "isolated"),
    },
    "quantity": {
        "positive": ("more", "many", "increase", "full", "multiple"),
        "negative": ("less", "few", "decrease", "empty", "none"),
    },
}

NEGATION_RE = re.compile(r"\b(?:not|never|no|without|isn['’]?t|aren['’]?t|doesn['’]?t|didn['’]?t)\b")
INTENSIFIERS = ("very", "extremely", "exceptionally", "deeply", "strongly", "really")
VALID_FAMILIES = set(RELATION_FAMILIES) | {"other"}
RELATION_FAMILY_ALIASES = {
    "amount": "quantity",
    "consumption": "pace",
    "count": "quantity",
    "credibility": "trust",
    "danger": "safety",
    "duration": "pace",
    "emotion": "sentiment",
    "evaluation": "sentiment",
    "group": "association",
    "isolation": "association",
    "mood": "sentiment",
    "performance": "outcome",
    "preference": "sentiment",
    "progress": "trajectory",
    "rate": "pace",
    "reliability": "trust",
    "result": "outcome",
    "security": "safety",
    "social": "association",
    "speed": "pace",
    "trend": "trajectory",
    "usage": "pace",
    "volume": "quantity",
}
STATE_STOPWORDS = {
    "that", "this", "with", "from", "into", "would", "could", "should",
    "their", "there", "being", "shown", "shows", "visual", "state",
    "person", "people", "object", "image", "scene", "visible", "appears",
}
ENTITY_STOPWORDS = STATE_STOPWORDS - {"person", "people", "object"}


def _normalize(value):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(value or "").lower()).split())


def _contains(text, cue):
    return bool(re.search(rf"\b{re.escape(cue)}\b", text))


def normalize_relation_family(value, *context):
    """Map descriptive model output to one supported relation family.

    Exact schema values always win. For a list of descriptors, contextual
    state language breaks ties so a pace claim containing a preference word
    is not incorrectly reduced to sentiment.
    """
    normalized = _normalize(value)
    if normalized in VALID_FAMILIES:
        return normalized

    terms = set(normalized.split())
    candidates = {
        RELATION_FAMILY_ALIASES[term]
        for term in terms
        if term in RELATION_FAMILY_ALIASES
    }
    context_text = _normalize(" ".join(str(item or "") for item in context))
    if "pace" in candidates and any(
        _contains(context_text, cue)
        for cue in ("quick", "quickly", "fast", "frequently", "rate", "week", "month", "last")
    ):
        return "pace"
    context_family, _, _ = _first_family_pole(context_text)
    if context_family in candidates:
        return context_family
    if len(candidates) == 1:
        return next(iter(candidates))
    return "other" if normalized == "other" else ""


def _first_family_pole(text):
    candidates = []
    for family, poles in RELATION_FAMILIES.items():
        for pole in ("positive", "negative"):
            for cue in poles[pole]:
                match = re.search(rf"\b{re.escape(cue)}\b", text)
                if match:
                    candidates.append((match.start(), family, pole, cue))
    if candidates:
        _, family, pole, cue = min(candidates)
        return family, pole, cue
    return None, None, None


def _cue_is_negated(text, cue):
    match = re.search(rf"\b{re.escape(cue)}\b", text)
    if not match:
        return False
    prefix = text[:match.start()].split()
    return bool(NEGATION_RE.search(" ".join(prefix[-3:])))


def _state_cues(value):
    return sorted({
        token for token in _normalize(value).split()
        if len(token) >= 4 and token not in STATE_STOPWORDS
    })


def _entity_cues(value):
    return sorted({
        token for token in _normalize(value).split()
        if len(token) >= 3 and token not in ENTITY_STOPWORDS
    })


def build_claim_relation(caption, language_output):
    language_output = language_output or {}
    source_caption = str(
        language_output.get("original_caption") or caption or ""
    ).strip()
    proposition = str(
        language_output.get("caption_proposition")
        or source_caption
        or ""
    ).strip()
    intended_proposition = str(
        language_output.get("intended_meaning") or proposition
    ).strip()
    contract = language_output.get("claim_contract") or audit_claim_contract(
        source_caption, language_output
    )
    normalized = _normalize(intended_proposition)
    family, pole, cue = _first_family_pole(normalized)
    negated = bool(cue and _cue_is_negated(normalized, cue))
    if pole and negated:
        pole = "negative" if pole == "positive" else "positive"
    declared_polarity = _normalize(
        language_output.get("caption_polarity", "")
    )
    if not pole and declared_polarity in {"positive", "negative"}:
        pole = declared_polarity
    expected_state = str(language_output.get("expected_visual_state", "")).strip()
    opposite_state = str(language_output.get("opposite_visual_state", "")).strip()
    parsed_family = normalize_relation_family(
        language_output.get("relation_family", ""),
        proposition,
        expected_state,
        opposite_state,
    )
    if parsed_family in VALID_FAMILIES and parsed_family != "other" and expected_state and opposite_state:
        family = parsed_family
    expected = list(RELATION_FAMILIES.get(family, {}).get(pole, ())) if pole else []
    opposite_pole = (
        "negative" if pole == "positive"
        else "positive" if pole == "negative"
        else None
    )
    opposite = list(RELATION_FAMILIES.get(family, {}).get(opposite_pole, ())) if opposite_pole else []
    if expected_state and expected_state.lower() not in {"none", "unknown"}:
        expected = sorted(set(expected) | set(_state_cues(expected_state)))
    if opposite_state and opposite_state.lower() not in {"none", "unknown"}:
        opposite = sorted(set(opposite) | set(_state_cues(opposite_state)))
    shared_cues = set(expected) & set(opposite)
    expected = [item for item in expected if item not in shared_cues]
    opposite = [item for item in opposite if item not in shared_cues]
    tokens = normalized.split()
    unresolved_reasons = list(contract.get("warnings", []) or [])
    if not family:
        unresolved_reasons.append("relation_family_unresolved")
    if (
        (not expected or not opposite)
        and not (expected_state and opposite_state)
    ):
        unresolved_reasons.append("directional_state_pair_unresolved")
    if not contract.get("safe_for_directional_reasoning", False):
        unresolved_reasons.append("claim_contract_not_direction_safe")
    return {
        "claim_text": source_caption,
        "caption_proposition": proposition,
        "intended_proposition": intended_proposition,
        "subject": str(language_output.get("claim_subject") or " ".join(tokens[: min(4, len(tokens))])),
        "predicate": str(language_output.get("claim_predicate") or cue or "unresolved"),
        "object": str(language_output.get("claim_object") or ""),
        "source": str(language_output.get("claim_source") or ""),
        "target": str(language_output.get("claim_target") or ""),
        "asserted_property": str(language_output.get("asserted_property") or ""),
        "transferred_property": str(language_output.get("transferred_property") or ""),
        "incongruity": str(language_output.get("incongruity") or ""),
        "caption_polarity": str(language_output.get("caption_polarity") or "unclear").lower(),
        "polarity": pole or "unresolved",
        "relation_family": family or "unresolved",
        "direction": pole or "unresolved",
        "intensity": "high" if any(_contains(normalized, item) for item in INTENSIFIERS) else "normal",
        "negated": negated,
        "expected_visual_cues": expected,
        "opposite_visual_cues": opposite,
        "figurative_mechanism": str(language_output.get("figurative_type", "unknown")).lower(),
        "expected_visual_state": expected_state,
        "opposite_visual_state": opposite_state,
        "claim_contract": contract,
        "resolved": bool(
            family
            and (
                (expected and opposite)
                or (expected_state and opposite_state)
            )
            and contract.get("safe_for_directional_reasoning", False)
        ),
        "unresolved_reasons": sorted(set(unresolved_reasons)),
    }


def attach_claim_relation(language_output, caption):
    output = deepcopy(language_output or {})
    output["claim_relation"] = build_claim_relation(caption, output)
    return output


def nominate_visual_relations(visual_output, claim_relation):
    if not (claim_relation or {}).get("resolved", False):
        return []
    facts = [
        *(visual_output.get("visual_facts", []) or []),
        *(visual_output.get("visual_relations", []) or []),
        *(visual_output.get("visible_text", []) or []),
    ]
    expected = claim_relation.get("expected_visual_cues", [])
    opposite = claim_relation.get("opposite_visual_cues", [])
    entity_cues = _entity_cues(" ".join(
        str(claim_relation.get(key, ""))
        for key in ("subject", "object", "source", "target")
    ))
    nominations = []
    seen = set()
    for fact in facts:
        normalized = _normalize(fact)
        entity_hits = [cue for cue in entity_cues if _contains(normalized, cue)]
        if entity_cues and not entity_hits:
            continue
        expected_hits = [
            cue for cue in expected
            if _contains(normalized, cue)
            and not _cue_is_negated(normalized, cue)
        ]
        opposite_hits = [
            cue for cue in opposite
            if _contains(normalized, cue)
            and not _cue_is_negated(normalized, cue)
        ]
        if expected_hits and not opposite_hits:
            relation, hits = "SUPPORT", expected_hits
        elif opposite_hits and not expected_hits:
            relation, hits = "CONFLICT", opposite_hits
        else:
            continue
        key = (" ".join(normalized.split()), relation)
        if key in seen:
            continue
        seen.add(key)
        nominations.append({
            "text": str(fact),
            "proposed_relation": relation,
            "relation_family": claim_relation.get("relation_family"),
            "claim_polarity": claim_relation.get("polarity"),
            "matched_cues": hits,
            "matched_entities": entity_hits,
            "method": "structured_relation_candidate",
        })
    return nominations
