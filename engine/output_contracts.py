"""Shared generation schemas; syntax constraints do not prove semantic truth."""
import json
import math
import re


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
NULLABLE_TEXT = {"type": ["string", "null"]}
TEXT_LIST = {"type": "array", "items": TEXT}
POLARITY = {"type": "string", "enum": ["positive", "negative", "neutral", "mixed", "unclear"]}
CORE = object_schema({
    **{key: TEXT for key in ("caption_proposition", "claim_subject", "claim_predicate",
                            "asserted_property", "expected_visual_state", "opposite_visual_state")},
    **{key: NULLABLE_TEXT for key in ("claim_object", "claim_source", "claim_target",
                                    "comparison_direction", "time_or_panel_scope")},
    "relation_family": {"type": "string", "enum": ["trajectory", "pace", "outcome", "sentiment",
        "safety", "trust", "association", "quantity", "other"]},
    "reasoning_requirement": {"type": "string", "enum": ["visual", "text_binding", "background", "normative", "mixed"]},
    "negation": TEXT_LIST, "quantities": TEXT_LIST, "claim_modifiers": TEXT_LIST,
})
INTERPRETATION = object_schema({
    "figurative_type": {"type": "string", "enum": ["sarcasm", "metaphor", "humor", "literal"]},
    "polarity_reversal": {"type": "string", "enum": ["yes", "no", "unclear"]},
    **{key: TEXT for key in ("linguistic_cue", "literal_meaning", "underlying_message",
                            "alternative_interpretation", "structural_reasoning_type", "background_knowledge")},
    "literal_polarity": POLARITY, "intended_polarity": POLARITY,
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
})


def validate_shape(value, schema):
    kind = schema.get("type")
    if isinstance(kind, list):
        return any(validate_shape(value, dict(schema, type=item)) for item in kind)
    if kind == "null":
        return value is None
    if kind == "string" and not isinstance(value, str):
        return False
    if kind == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
        return False
    if kind == "boolean" and type(value) is not bool:
        return False
    if kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            return False
    if kind == "array":
        return (isinstance(value, list)
                and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf)
                and all(validate_shape(item, schema["items"]) for item in value))
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["required"]):
            return False
        return all(validate_shape(value[key], definition) for key, definition in schema["properties"].items())
    return "enum" not in schema or value in schema["enum"]


def prefix_constraint(tokenizer, schema):
    from engine.structured_decoder import prefix_constraint as build
    return build(tokenizer, schema)


def incomplete_clause(text):
    """Conservative signal of an unfinished clause, not a fluency/truth classifier."""
    text = str(text or "").strip()
    if not text or len(text.split()) < 2:
        return False
    return bool(re.search(r"\b(?:a|an|the|because|although|whereas|which|whose|with|without|such as|due to|rather than)\s*$", text, re.I)
                or text.endswith((",", ";", ":", "(", "[")))


def saturated_text_fields(text, schema):
    """Report clear unfinished clauses; a missing final period is not truncation."""
    if not schema:
        return []
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(value, dict):
        return []
    return [key for key, definition in schema.get("properties", {}).items()
            if isinstance(value.get(key), str) and definition.get("maxLength")
            and incomplete_clause(value[key])]


def complete_json_stopper(tokenizer, prompt_length, schema):
    """Stop after the first complete schema-valid object, not trailing padding."""
    from transformers import StoppingCriteria

    class CompleteJSON(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if len(input_ids) != 1:
                raise ValueError("Structured generation currently requires batch size one")
            text = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
            try:
                return validate_shape(json.loads(text), schema)
            except (ValueError, TypeError):
                return False
    return CompleteJSON()


def schema_for_contract(name):
    relation = {"type": "string", "enum": ["SUPPORT", "CONFLICT", "UNRESOLVED"]}
    if name == "independent_candidate":
        return object_schema({"relation": relation, **{key: TEXT for key in (
            "claim_reading", "decisive_observation", "alternative", "decisive_question")}})
    if name == "independent_verification":
        return object_schema({"relation": relation, **{key: {"type": "boolean"} for key in (
            "visual_premise_supported", "caption_premise_preserved", "entity_scope_consistent")},
            "evidence_ids": TEXT_LIST, "reason": TEXT})
    if name == "tribunal_review":
        # Bound the whole generation, not individual natural-language clauses.
        concise = {"type": "string"}
        from engine.semantic_bridge import BRIDGE_FAMILIES
        return object_schema({
            "relation": relation,
            "admissibility": {"type": "string", "enum": ["VERIFIED", "CORROBORATED", "PLAUSIBLE", "INSUFFICIENT"]},
            "node_relations": {"type": "array", "maxItems": 8, "items": object_schema({
                "claim_node_id": TEXT, "relation": relation, "evidence_ids": TEXT_LIST})},
            "semantic_bridge_type": {"type": "string", "enum": sorted(BRIDGE_FAMILIES)},
            **{key: concise for key in ("visual_premise", "caption_premise",
                                    "semantic_bridge", "counter_interpretation", "reason")},
            "evidence_ids": {"type": "array", "maxItems": 4,
                             "items": {"type": "string", "maxLength": 32}},
            "context_requests": {"type": "array", "maxItems": 4,
                                 "items": {"type": "string", "maxLength": 32}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "counter_interpretation_strength": {"type": "number", "minimum": 0, "maximum": 1},
            "requested_follow_up": {"type": "string", "enum": ["NONE", "VISUAL_PREMISE", "CAPTION_PREMISE",
                "ENTITY_BINDING", "SCOPE_BINDING", "COUNTER_INTERPRETATION"]},
        })
    return None
