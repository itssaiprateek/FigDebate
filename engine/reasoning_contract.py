"""Completion and grounded-reading contracts; structure never proves semantic truth."""
import re
from engine.output_contracts import incomplete_clause, object_schema

VERSION = "grounded-reading-1"
RULES = (
    "Resolve the source expression against the depicted referent before judging it. "
    "For a metaphor, analogy or joke, identify the compared property and the specific observations "
    "that license that comparison. Physical presence of the metaphor's source object is unnecessary. "
    "For literal text, retain the actual asserted property. For sarcasm, distinguish expressed evaluation "
    "from speaker attitude; do not reverse the claim silently. Mere subject/topic overlap is insufficient. "
    "Preserve qualifiers, roles, negation and scope. A missing or ambiguous connection stays UNRESOLVED. "
    "A figurative hypothesis is not a new visual observation or verified evidence."
)
ASSESSMENT_FIELDS = (
    "Use these eight short lines, with one complete sentence for Relation Analysis. "
    "Use UNRESOLVED for an unidentified referent or property, not an invented completion.\n"
    "Visual Evidence: the smallest sufficient set of supplied observations\n"
    "Evidence IDs: supplied IDs only, or NONE\n"
    "Expression: exact source-caption expression\n"
    "Referent: depicted participant/object/reaction being described\n"
    "Property: compared or asserted property, including necessary qualifiers\n"
    "Caption Meaning: full source assertion under this reading\n"
    "Relation Status: SUPPORT, CONFLICT, or UNRESOLVED\n"
    "Relation Analysis: connect the cited observations to that property and explain the relation\n"
)
READING_SCHEMA = object_schema({
    "reading": {"type": "string", "enum": ["LITERAL", "FIGURATIVE", "UNRESOLVED"]},
    **{k: {"type": "string", "minLength": 1} for k in ("expression", "referent", "property", "reason")},
    "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}},
})


def fields(text):
    names = "Visual Evidence|Evidence IDs|Expression|Referent|Property|Caption Meaning|Relation Status|Relation Analysis"
    return {m.group(1): m.group(2).strip() for m in re.finditer(
        rf"(?m)^({names}):\s*([\s\S]*?)(?=^(?:{names}):|\Z)", str(text))}


def completion_status(text, diagnostics=None, *, grounded=False, source=None):
    diagnostics = diagnostics or {}
    parsed = fields(text)
    problems = []
    if diagnostics.get("input_truncated"):
        problems.append("input_truncated")
    if diagnostics.get("hit_token_limit"):
        problems.append("generation_limit")
    required = ["Visual Evidence", "Evidence IDs", "Caption Meaning", "Relation Analysis"]
    if grounded:
        required += ["Expression", "Referent", "Property", "Relation Status"]
    problems += ["missing:" + key for key in required if not parsed.get(key)]
    reason = parsed.get("Relation Analysis", "")
    if reason and incomplete_clause(reason, conjunctions=True):
        problems.append("unfinished_relation_analysis")
    expression = parsed.get("Expression", "")
    if len(expression) > 1 and expression[0] == expression[-1] and expression[0] in "\"'":
        expression = expression[1:-1]
    if grounded and source is not None and (not expression or expression not in source):
        problems.append("expression_must_quote_original_caption_not_witness_definition")
    if grounded and parsed.get("Relation Status") not in {"SUPPORT", "CONFLICT", "UNRESOLVED"}:
        problems.append("invalid_relation_status")
    if grounded and parsed.get("Relation Status") in {"SUPPORT", "CONFLICT"} and any(
            parsed.get(key, "").strip().lower() in {"none", "unknown", "unresolved", ""} for key in ("Referent", "Property")):
        problems.append("direction_requires_resolved_referent_and_property")
    return {"complete": not problems, "issues": problems, "fields": parsed,
            "semantic_correctness": "NOT_ESTABLISHED_BY_COMPLETENESS"}


def reading_error(value, source, known):
    if value["expression"] not in source:
        return "expression must quote the source caption exactly"
    if not set(value["evidence_ids"]) <= set(known):
        return "Cite only selected observations"
    if incomplete_clause(value["reason"], conjunctions=True):
        return "Finish the observation-to-property explanation"
    if value["reading"] != "UNRESOLVED" and any(value[k].strip().lower() in {"none", "unknown", "unresolved", ""} for k in ("referent", "property")):
        return "An unresolved referent or property requires UNRESOLVED reading"


def final_explanation_status(text, decision):
    if decision.get("_explanation_generation_failed"):
        return {"complete": False, "issues": ["bounded_generation_failed"], "method": "generation_audit"}
    recorded = decision.get("_assessment_completion")
    if recorded and text == decision.get("_arbiter_assessment"):
        return {"complete": recorded["complete"], "issues": recorded["issues"], "method": "generation_and_field_contract"}
    complete = bool(str(text).strip()) and not incomplete_clause(text, conjunctions=True)
    return {"complete": complete, "issues": [] if complete else ["empty_or_unfinished_explanation"],
            "method": "surface_check_only", "semantic_correctness": "NOT_ESTABLISHED"}


def grounded_language_summary(language):
    """Keep linguistic hypotheses separate from raw caption and visual observations.

    The literal dictionary gloss is retained in the witness trace, not repeated
    as the proposition to prove. The unmodified caption supplies that proposition.
    """
    import json
    candidates = {}
    for key in ("figurative_type", "background_knowledge", "intended_meaning", "alternative_interpretation"):
        value = language.get(key)
        if value and str(value).strip().lower() not in {"none", "unknown", "not specified"}:
            candidates[key] = value
    return ("Fallible caption-only hypotheses, not visual evidence or required literal events. "
            "Resolve their applicability jointly from the original caption and image observations.\n"
            + json.dumps(candidates, ensure_ascii=False))


def assessment_schema(known):
    return object_schema({
        "evidence_ids": {"type": "array", "maxItems": 4, "items": {"type": "string", **({"enum": sorted(known)} if known else {})}},
        **{key: {"type": "string", "minLength": 1} for key in ("referent", "property", "meaning")},
        "relation": {"type": "string", "enum": ["SUPPORT", "CONFLICT", "UNRESOLVED"]},
        "reason": {"type": "string", "minLength": 1},
    })


def render_assessment(raw, source, catalog):
    """Copy the source and cited observations; generate only the interpretation."""
    import json
    from engine.output_contracts import validate_shape
    by_id = {item["id"]: item.get("text", "") for item in catalog if item.get("id")}
    value = json.loads(raw)
    if not validate_shape(value, assessment_schema(by_id)) or not set(value['evidence_ids']) <= set(by_id):
        raise ValueError("Invalid structured assessment")
    if len(set(value['evidence_ids'])) != len(value['evidence_ids']):
        raise ValueError("Duplicate observation IDs")
    observations = " ".join(f"[{key}] {by_id[key]}" for key in value['evidence_ids'])
    return (f"Visual Evidence: {observations or 'No supplied observation selected.'}\n"
            f"Evidence IDs: {', '.join(value['evidence_ids']) or 'NONE'}\n"
            f"Expression: {source}\nReferent: {value['referent']}\nProperty: {value['property']}\n"
            f"Caption Meaning: {value['meaning']}\nRelation Status: {value['relation']}\n"
            f"Relation Analysis: {value['reason']}")
