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

FIELD_HEADING_RE = re.compile(
    r"\b(?:caption proposition|claim subject|claim predicate|claim object|"
    r"claim source|claim target|asserted property|expected visual state|"
    r"opposite visual state|reasoning requirement|background knowledge|"
    r"structural reasoning type|literal polarity|intended polarity|"
    r"comparison direction|evaluation target|time or panel scope)\s*:",
    flags=re.IGNORECASE,
)
GENERIC_PREDICATES = {
    "is", "are", "was", "were", "be", "being", "exists", "happened",
    "happens", "occurred", "occurs", "took place",
}
ABSENCE_PREFIX_RE = re.compile(
    r"^(?:no\b|not visible\b|nothing\b|none\b|absence\b|absent\b|"
    r"missing\b|without\b|there (?:is|are) no\b)",
    flags=re.IGNORECASE,
)

# General English state oppositions used only to validate whether Agent 2 has
# supplied a genuinely directional pair. These are semantic relation classes,
# never image-, dataset-, or sample-specific rules.
STATE_OPPOSITION_GROUPS = (
    ({"up", "upward", "rise", "rising", "increase", "growth", "recover"},
     {"down", "downward", "fall", "falling", "decrease", "decline", "collapse"}),
    ({"slow", "slowly", "leisurely", "calm", "relaxed", "serene"},
     {"fast", "quickly", "rushing", "agitated", "tense", "chaotic"}),
    ({"whole", "intact", "healthy", "working", "successful", "success"},
     {"broken", "damaged", "rotten", "failed", "failure", "useless"}),
    ({"present", "visible", "attached", "open", "using", "used"},
     {"absent", "missing", "detached", "closed", "unused", "abandoned"}),
    ({"safe", "respectful", "honest", "truthful", "positive"},
     {"unsafe", "disrespectful", "dishonest", "false", "negative"}),
    ({"happy", "happiness", "joy", "joyful", "content", "elated",
      "celebratory", "pleased", "hopeful"},
     {"sad", "sadness", "angry", "anger", "distressed", "devastating",
      "disappointed", "disappointment", "awful", "worried", "fearful"}),
    ({"more", "greater", "many", "frequent"},
     {"less", "fewer", "few", "infrequent", "equal"}),
    ({"long", "lasting", "persistent", "durable"},
     {"short", "brief", "temporary", "fleeting"}),
    ({"enough", "sufficient", "adequate"},
     {"insufficient", "inadequate", "lacking"}),
    ({"expected", "support", "supports"},
     {"opposite", "conflict", "conflicts"}),
)

STATE_STOPWORDS = ENTITY_STOPWORDS | {
    "appears", "condition", "displayed", "explicitly", "matches",
    "observable", "observed", "show", "shown", "shows", "state",
    "visible", "visibly",
}

NORMATIVE_CUES = {
    "abusive", "cruel", "demeaning", "disrespectful", "ethical", "fair",
    "gross", "immoral", "inappropriate", "misogynistic", "offensive",
    "respectful", "rude", "sexist", "unethical", "unfair",
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
    # Keep source auditing lexical and domain-neutral.  Semantic aliases must
    # come from the generated claim frame and be verified downstream; a fixed
    # vocabulary here can silently make one dataset example look valid.
    return set(_normalize(value).split())


def _english_forms(token):
    """Return conservative surface/base forms for common English inflections.

    The contract only uses these forms to recognize members of an explicit
    semantic opposition group.  They are not used to make entities equivalent
    or to introduce domain aliases.
    """
    forms = {token}
    if len(token) > 4 and token.endswith("ies"):
        forms.add(token[:-3] + "y")
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        forms.add(token[:-1])
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        forms.update({stem, stem + "e"})
        if len(stem) > 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        forms.update({stem, stem + "e"})
        if len(stem) > 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    return forms


def _opposition_forms(tokens):
    return {
        form
        for token in tokens
        for form in _english_forms(token)
    }


def _state_clauses(value):
    return [
        frozenset(_state_tokens(part))
        for part in re.split(
            r"\s*(?:;|\||\bwhile\b|\bwhereas\b|\bbut\b)\s*",
            str(value or ""),
            flags=re.IGNORECASE,
        )
        if _state_tokens(part)
    ]


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


def _state_tokens(value):
    return {
        token for token in _tokens(value)
        if len(token) >= 3 and token not in STATE_STOPWORDS
    }


def _background_required(language_output):
    value = language_output.get("background_knowledge", "")
    return not _is_placeholder(value) and _normalize(value) not in {
        "not specified", "not required"
    }


def _normative_required(language_output):
    declared = _normalize(language_output.get("reasoning_requirement", ""))
    if declared in {"normative", "mixed"}:
        return True
    text = " ".join(
        _normalize(language_output.get(key, ""))
        for key in (
            "caption_proposition", "asserted_property", "intended_meaning"
        )
    )
    return bool(_tokens(text) & NORMATIVE_CUES)


def _field_contamination(language_output):
    contaminated = []
    for key in (
        "caption_proposition", "claim_subject", "claim_predicate",
        "claim_object", "claim_source", "claim_target",
        "asserted_property", "expected_visual_state",
        "opposite_visual_state", "comparison_direction",
        "evaluation_target", "time_or_panel_scope",
    ):
        if FIELD_HEADING_RE.search(str(language_output.get(key) or "")):
            contaminated.append(key)
    return contaminated


def _affirmative_opposite(value):
    normalized = " ".join(str(value or "").split()).strip()
    return bool(normalized) and not ABSENCE_PREFIX_RE.match(normalized)


def audit_relation_pair(expected_state, opposite_state, subject=""):
    """Conservatively validate a generated support/conflict state pair."""
    expected = _normalize(expected_state)
    opposite = _normalize(opposite_state)
    expected_tokens = _state_tokens(expected)
    opposite_tokens = _state_tokens(opposite)
    expected_forms = _opposition_forms(expected_tokens)
    opposite_forms = _opposition_forms(opposite_tokens)
    subject_tokens = _entity_tokens(subject)
    warnings = []
    if not expected or not opposite:
        warnings.append("directional_state_pair_incomplete")
    if expected and expected == opposite:
        warnings.append("directional_state_pair_identical")

    opposition_signals = []
    for positive, negative in STATE_OPPOSITION_GROUPS:
        if (
            expected_forms & positive and opposite_forms & negative
        ) or (
            expected_forms & negative and opposite_forms & positive
        ):
            signals = (
                expected_forms & (positive | negative)
            ) | (
                opposite_forms & (positive | negative)
            )
            opposition_signals.append("/".join(sorted(signals)))
    # Negation is directional only when both sides negate the same underlying
    # condition.  Merely placing "without" in an unrelated clause must not
    # make two otherwise different states look like opposites.
    shared_negated_condition = bool(
        expected_forms & opposite_forms
        or expected_tokens & opposite_tokens
    )
    negation_opposition = bool(
        shared_negated_condition
        and (
            (_negations(expected) and not _negations(opposite))
            or (_negations(opposite) and not _negations(expected))
        )
    )
    if negation_opposition:
        opposition_signals.append("explicit_negation")

    # A multi-entity relation may express its opposite by swapping outcomes
    # rather than using antonyms: "A has X; B has Y" versus
    # "A has Y; B has X".  Accept only a true token-preserving reassignment;
    # merely reordering the same clauses does not qualify.
    expected_clauses = _state_clauses(expected_state)
    opposite_clauses = _state_clauses(opposite_state)
    if len(expected_clauses) >= 2 and len(opposite_clauses) >= 2:
        expected_union = set().union(*expected_clauses)
        opposite_union = set().union(*opposite_clauses)
        if (
            expected_union == opposite_union
            and set(expected_clauses) != set(opposite_clauses)
        ):
            opposition_signals.append("structured_relation_reassignment")

    shared_state_tokens = (
        expected_tokens & opposite_tokens
    ) - subject_tokens
    # Expected/opposite are typed fields already attached to `subject`.  They
    # need not redundantly repeat that subject ("rotten" versus "healthy").
    # When no subject is available, retain the older shared-anchor safeguard.
    anchor_preserved = bool(subject_tokens or shared_state_tokens)
    if not anchor_preserved:
        warnings.append("directional_state_pair_lacks_shared_anchor")
    if expected and opposite and not opposition_signals:
        warnings.append("directional_state_pair_not_proven_opposing")

    return {
        "complete": bool(expected and opposite),
        "distinct": bool(expected and opposite and expected != opposite),
        "anchor_preserved": anchor_preserved,
        "opposition_signals": sorted(set(opposition_signals)),
        "valid": bool(
            expected and opposite and expected != opposite
            and anchor_preserved and opposition_signals
        ),
        "warnings": warnings,
    }


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
    contaminated_fields = _field_contamination(language_output)
    if contaminated_fields:
        warnings.extend(
            f"field_heading_contamination:{field}"
            for field in contaminated_fields
        )
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
    relation_pair = audit_relation_pair(
        language_output.get("expected_visual_state"),
        language_output.get("opposite_visual_state"),
        entity_fields.get("claim_subject", ""),
    )
    relation_pair_complete = relation_pair["complete"]
    opposite_is_affirmative = _affirmative_opposite(
        language_output.get("opposite_visual_state")
    )
    if relation_pair_complete and not opposite_is_affirmative:
        warnings.append("opposite_state_is_absence_only")
    predicate = _normalize(language_output.get("claim_predicate", ""))
    predicate_specific = None if not predicate else predicate not in GENERIC_PREDICATES
    if predicate_specific is False:
        warnings.append("claim_predicate_too_generic")
    background_required = _background_required(language_output)
    normative_required = _normative_required(language_output)
    declared_reasoning = _normalize(
        language_output.get("reasoning_requirement", "")
    )
    if declared_reasoning not in {
        "visual", "text binding", "text_binding", "background",
        "normative", "mixed",
    }:
        declared_reasoning = "mixed" if (
            background_required or normative_required
        ) else "visual"
    source_safe = bool(
        proposition_preserved
        and entity_frame_preserved
        and relation_pair_complete
        and relation_pair["valid"]
        and not contaminated_fields
        and predicate_specific is not False
        and opposite_is_affirmative
    )
    tribunal_safe = bool(
        proposition_preserved
        and entity_frame_preserved
        and relation_pair_complete
        and relation_pair["distinct"]
        and relation_pair["anchor_preserved"]
        and not contaminated_fields
        and predicate_specific is not False
        and opposite_is_affirmative
    )
    return {
        "schema_version": "3.0",
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
        "relation_pair_valid": relation_pair["valid"],
        "relation_pair_audit": relation_pair,
        "opposite_state_is_affirmative": opposite_is_affirmative,
        "predicate_specific": predicate_specific,
        "contaminated_fields": contaminated_fields,
        "structural_reasoning_type": str(
            language_output.get("structural_reasoning_type") or "UNRESOLVED"
        ),
        "figurative_mechanism_candidates": language_output.get(
            "figurative_mechanism_candidates", []
        ),
        "literal_polarity": str(
            language_output.get("literal_polarity") or "unclear"
        ),
        "intended_polarity": str(
            language_output.get("intended_polarity") or "unclear"
        ),
        "comparison_direction": str(
            language_output.get("comparison_direction") or ""
        ),
        "evaluation_target": str(
            language_output.get("evaluation_target") or ""
        ),
        "time_or_panel_scope": str(
            language_output.get("time_or_panel_scope") or ""
        ),
        "reasoning_requirement": declared_reasoning,
        "requires_background_knowledge": background_required,
        "requires_normative_reasoning": normative_required,
        "safe_for_directional_reasoning": source_safe,
        "safe_for_tribunal_reasoning": tribunal_safe,
        "safe_for_automatic_directional_reasoning": bool(
            source_safe and not background_required and not normative_required
        ),
        "warnings": sorted(set(warnings + relation_pair["warnings"])),
    }


def attach_claim_contract(language_output, caption):
    output = deepcopy(language_output or {})
    output["original_caption"] = " ".join(str(caption or "").split())
    output["claim_contract"] = audit_claim_contract(caption, output)
    return output
