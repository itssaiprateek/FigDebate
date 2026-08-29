"""Typed, label-blind reasoning routes shared by every figurative mechanism.

The dataset phenomenon is deliberately not an input to this module.  A route
is inferred from the immutable caption analysis and the available evidence so
the same code can be used for metaphor, sarcasm, humour, and literal examples
without collapsing their different inference requirements.
"""

from __future__ import annotations

from copy import deepcopy
import re


STRUCTURAL_TYPES = {
    "DIRECT_STATE",
    "OCR_REGION_BINDING",
    "COMPARATIVE_LAYOUT",
    "TEMPORAL_CAUSAL_SEQUENCE",
    "QUOTED_STATEMENT_AND_REACTION",
    "AFFECTIVE_SCENE",
    "SYMBOL_ATTACHMENT",
    "BACKGROUND_REQUIRED",
    "UNRESOLVED",
}

FIGURATIVE_MECHANISMS = {
    "LITERAL",
    "METAPHOR_MAPPING",
    "SARCASM_POLARITY",
    "HUMOR_INCONGRUITY",
    "UNRESOLVED",
}

_STRUCTURAL_ALIASES = {
    "direct": "DIRECT_STATE",
    "direct state": "DIRECT_STATE",
    "visual": "DIRECT_STATE",
    "ocr": "OCR_REGION_BINDING",
    "text binding": "OCR_REGION_BINDING",
    "text_binding": "OCR_REGION_BINDING",
    "layout": "OCR_REGION_BINDING",
    "comparison": "COMPARATIVE_LAYOUT",
    "comparative": "COMPARATIVE_LAYOUT",
    "comparison layout": "COMPARATIVE_LAYOUT",
    "sequence": "TEMPORAL_CAUSAL_SEQUENCE",
    "temporal": "TEMPORAL_CAUSAL_SEQUENCE",
    "causal": "TEMPORAL_CAUSAL_SEQUENCE",
    "reaction": "QUOTED_STATEMENT_AND_REACTION",
    "quoted statement": "QUOTED_STATEMENT_AND_REACTION",
    "affect": "AFFECTIVE_SCENE",
    "emotion": "AFFECTIVE_SCENE",
    "sentiment": "AFFECTIVE_SCENE",
    "symbol": "SYMBOL_ATTACHMENT",
    "background": "BACKGROUND_REQUIRED",
}

_MECHANISM_ALIASES = {
    "literal": "LITERAL",
    "metaphor": "METAPHOR_MAPPING",
    "metaphorical": "METAPHOR_MAPPING",
    "sarcasm": "SARCASM_POLARITY",
    "sarcastic": "SARCASM_POLARITY",
    "irony": "SARCASM_POLARITY",
    "ironic": "SARCASM_POLARITY",
    "humor": "HUMOR_INCONGRUITY",
    "humour": "HUMOR_INCONGRUITY",
    "funny": "HUMOR_INCONGRUITY",
    "incongruity": "HUMOR_INCONGRUITY",
}


def _normalize(value: object) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9_ ]", " ", str(value or "").casefold()).split()
    )


def normalize_structural_type(value: object) -> str:
    normalized = _normalize(value)
    canonical = normalized.upper().replace(" ", "_")
    if canonical in STRUCTURAL_TYPES:
        return canonical
    return _STRUCTURAL_ALIASES.get(normalized, "UNRESOLVED")


def normalize_figurative_mechanism(value: object) -> str:
    normalized = _normalize(value)
    canonical = normalized.upper().replace(" ", "_")
    if canonical in FIGURATIVE_MECHANISMS:
        return canonical
    return _MECHANISM_ALIASES.get(normalized, "UNRESOLVED")


def mechanism_candidates(language_output: dict) -> list[str]:
    """Return ordered mechanism candidates without using a dataset label."""
    language_output = language_output or {}
    declared = language_output.get("figurative_mechanism_candidates", [])
    if isinstance(declared, str):
        declared = re.split(r"[,|;/]", declared)
    candidates = [
        normalize_figurative_mechanism(item) for item in (declared or [])
    ]

    primary = normalize_figurative_mechanism(
        language_output.get("figurative_type", "")
    )
    polarity_text = _normalize(
        " ".join(str(language_output.get(key, "")) for key in (
            "polarity_reversal", "literal_polarity", "intended_polarity",
            "linguistic_cue",
        ))
    )
    if any(token in polarity_text.split() for token in ("yes", "reverse", "reversal")):
        candidates.insert(0, "SARCASM_POLARITY")
    if language_output.get("transferred_property"):
        candidates.insert(0, "METAPHOR_MAPPING")
    if language_output.get("incongruity"):
        candidates.append("HUMOR_INCONGRUITY")
    candidates.append(primary)

    ordered = []
    for candidate in candidates:
        if candidate in FIGURATIVE_MECHANISMS and candidate not in ordered:
            ordered.append(candidate)
    if not ordered or ordered == ["UNRESOLVED"]:
        return ["UNRESOLVED"]
    return [item for item in ordered if item != "UNRESOLVED"][:2]


def infer_structural_type(language_output: dict, comparison: dict | None = None) -> str:
    """Infer the evidence operation independently from figurative style."""
    language_output = language_output or {}
    comparison = comparison or {}
    declared = normalize_structural_type(
        language_output.get("structural_reasoning_type", "")
    )
    if declared != "UNRESOLVED":
        return declared

    contract = language_output.get("claim_contract", {}) or {}
    relation = comparison.get("claim_relation", {}) or language_output.get(
        "claim_relation", {}
    ) or {}
    family = _normalize(
        relation.get("relation_family") or language_output.get("relation_family")
    )
    text = _normalize(" ".join(str(language_output.get(key, "")) for key in (
        "caption_proposition", "asserted_property", "incongruity",
        "expected_visual_state", "opposite_visual_state",
    )))

    if contract.get("requires_background_knowledge", False):
        return "BACKGROUND_REQUIRED"
    if comparison.get("relation_binding_required", False):
        return "OCR_REGION_BINDING"
    if language_output.get("comparison_direction") or family in {
        "pace", "quantity", "outcome"
    } or any(token in text.split() for token in (
        "compare", "comparison", "more", "less", "fewer", "than",
    )):
        return "COMPARATIVE_LAYOUT"
    if any(token in text.split() for token in (
        "before", "after", "causes", "caused", "because", "blames",
        "sequence", "result",
    )):
        return "TEMPORAL_CAUSAL_SEQUENCE"
    if language_output.get("evaluation_target") and any(
        token in text.split() for token in ("reaction", "responds", "quote", "headline")
    ):
        return "QUOTED_STATEMENT_AND_REACTION"
    if family == "sentiment":
        return "AFFECTIVE_SCENE"
    if comparison.get("has_symbolic_evidence", False):
        return "SYMBOL_ATTACHMENT"
    return "DIRECT_STATE"


def attach_reasoning_profile(
    language_output: dict,
    comparison: dict | None = None,
) -> dict:
    output = deepcopy(language_output or {})
    candidates = mechanism_candidates(output)
    output["figurative_mechanism_candidates"] = candidates
    output["structural_reasoning_type"] = infer_structural_type(
        output, comparison
    )
    if isinstance(output.get("claim_contract"), dict):
        contract = deepcopy(output["claim_contract"])
        contract["structural_reasoning_type"] = output[
            "structural_reasoning_type"
        ]
        contract["figurative_mechanism_candidates"] = list(candidates)
        for field in (
            "literal_polarity", "intended_polarity", "comparison_direction",
            "evaluation_target", "time_or_panel_scope",
        ):
            contract[field] = output.get(field, contract.get(field, ""))
        output["claim_contract"] = contract
    output["reasoning_profile"] = {
        "schema_version": "1.0",
        "structural_type": output["structural_reasoning_type"],
        "figurative_mechanisms": candidates,
        "primary_figurative_mechanism": candidates[0],
        "uses_gold_phenomenon": False,
    }
    return output
