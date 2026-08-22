"""Immutable caption and claim-frame validation for FigDebate.

Agent 2 may interpret a caption, but it must not silently replace the caption's
entities, numbers, or explicit negation.  This module keeps the source caption
separate from generated interpretations and records conservative validation
signals used by downstream evidence routing.
"""

from copy import deepcopy
import re


NEGATION_TOKENS = {
    "no", "not", "never", "none", "neither", "nor", "without",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "weren't", "werent", "doesn't", "doesnt", "didn't", "didnt",
    "can't", "cant", "cannot", "won't", "wont",
    "ain't", "aint",
}
ENTITY_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being",
    "caption", "could", "does", "from", "have", "image", "into", "just",
    "more", "most", "none", "only", "other", "should", "some", "that",
    "their", "there", "these", "they", "this", "those", "through", "under",
    "very", "what", "when", "where", "which", "while", "with", "would",
    "same", "subject", "object", "source", "target", "the",
}
PLACEHOLDER_VALUES = {
    "", "none", "n a", "na", "not applicable", "not specified",
    "unknown", "unspecified", "unclear", "implicit", "implicitly",
}
TOKEN_ALIASES = {
    "anything": "object",
    "good": "product",
    "goods": "product",
    "item": "product",
    "items": "product",
    "products": "product",
    "disliked": "dislike",
    "dislikes": "dislike",
    "hated": "dislike",
    "hates": "dislike",
    "hate": "dislike",
    "liked": "love",
    "likes": "love",
    "loved": "love",
    "loves": "love",
    "favorite": "love",
    "favourite": "love",
}
HUMAN_REFERENCE_TOKENS = {
    "he", "her", "hers", "him", "his", "human", "man", "men",
    "person", "people", "she", "someone", "speaker", "they", "woman",
    "women",
}
SOURCE_HUMAN_PRONOUNS = {
    "he", "her", "hers", "him", "his", "she", "their", "theirs", "they",
}


def _normalize(value):
    return " ".join(
        re.sub(r"[^a-z0-9' ]", " ", str(value or "").casefold()).split()
    )


def _tokens(value):
    return {
        TOKEN_ALIASES.get(token, token)
        for token in _normalize(value).split()
    }


def _numbers(value):
    return sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", str(value or ""))))


def _negations(value):
    return sorted(_tokens(value) & NEGATION_TOKENS)


def _entity_tokens(value):
    return {
        token for token in _tokens(value)
        if len(token) >= 3 and token not in ENTITY_STOPWORDS
    }


def _is_placeholder(value):
    normalized = _normalize(value)
    if normalized in PLACEHOLDER_VALUES:
        return True
    if normalized.startswith(("unknown ", "unspecified ", "unclear ")):
        return True
    return bool(normalized) and all(
        token in PLACEHOLDER_VALUES
        for token in normalized.replace("implicitly", "implicit").split()
    )


def _has_human_coreference(source_value, entity_value):
    source = _tokens(source_value)
    entity = _tokens(entity_value)
    return bool(source & SOURCE_HUMAN_PRONOUNS) and bool(
        entity & HUMAN_REFERENCE_TOKENS
    )


def audit_claim_contract(caption, language_output):
    """Audit preservation without pretending to solve semantic equivalence."""
    language_output = language_output or {}
    source_caption = " ".join(str(caption or "").split())
    proposition = " ".join(
        str(language_output.get("caption_proposition") or "").split()
    )
    source_numbers = _numbers(source_caption)
    proposition_numbers = _numbers(proposition)
    source_negations = _negations(source_caption)
    proposition_negations = _negations(proposition)
    source_proposition_tokens = _entity_tokens(source_caption)
    generated_proposition_tokens = _entity_tokens(proposition)

    entity_fields = {
        key: " ".join(str(language_output.get(key) or "").split())
        for key in (
            "claim_subject", "claim_object", "claim_source", "claim_target"
        )
    }
    if _normalize(entity_fields["claim_source"]) in {
        "same as subject", "subject",
    }:
        entity_fields["claim_source"] = entity_fields["claim_subject"]
    if _normalize(entity_fields["claim_target"]) in {
        "same as object", "object", "same as subject", "subject",
    }:
        entity_fields["claim_target"] = (
            entity_fields["claim_object"]
            if _normalize(entity_fields["claim_target"]) in {
                "same as object", "object",
            }
            else entity_fields["claim_subject"]
        )
    source_tokens = _entity_tokens(source_caption)
    entity_checks = {}
    for key, value in entity_fields.items():
        if _is_placeholder(value):
            entity_checks[key] = None
            continue
        value_tokens = _entity_tokens(value)
        entity_checks[key] = (
            None
            if not value_tokens
            else bool(source_tokens & value_tokens)
            or _has_human_coreference(source_caption, value)
        )

    warnings = []
    if not proposition:
        warnings.append("missing_caption_proposition")
    if source_numbers and not set(source_numbers).issubset(proposition_numbers):
        warnings.append("caption_number_changed_or_dropped")
    if source_negations and not proposition_negations:
        warnings.append("caption_negation_changed_or_dropped")
    if (
        source_proposition_tokens
        and not source_proposition_tokens.intersection(
            generated_proposition_tokens
        )
    ):
        warnings.append("caption_proposition_has_no_source_anchor")
    for key, preserved in entity_checks.items():
        if preserved is False:
            warnings.append(f"{key}_not_grounded_in_caption")

    required_entity_checks = [
        preserved for preserved in entity_checks.values()
        if preserved is not None
    ]
    entity_frame_preserved = (
        all(required_entity_checks) if required_entity_checks else False
    )
    proposition_preserved = not any(
        warning in {
            "missing_caption_proposition",
            "caption_number_changed_or_dropped",
            "caption_negation_changed_or_dropped",
            "caption_proposition_has_no_source_anchor",
        }
        for warning in warnings
    )
    relation_pair_complete = bool(
        str(language_output.get("expected_visual_state") or "").strip()
        and str(language_output.get("opposite_visual_state") or "").strip()
    )
    return {
        "source_caption": source_caption,
        "caption_proposition": proposition,
        "source_numbers": source_numbers,
        "proposition_numbers": proposition_numbers,
        "source_negations": source_negations,
        "proposition_negations": proposition_negations,
        "entity_checks": entity_checks,
        "entity_frame_preserved": entity_frame_preserved,
        "proposition_preserved": proposition_preserved,
        "relation_pair_complete": relation_pair_complete,
        "safe_for_directional_reasoning": bool(
            proposition_preserved
            and entity_frame_preserved
            and relation_pair_complete
        ),
        "warnings": warnings,
    }


def attach_claim_contract(language_output, caption):
    output = deepcopy(language_output or {})
    output["original_caption"] = " ".join(str(caption or "").split())
    output["claim_contract"] = audit_claim_contract(caption, output)
    return output
