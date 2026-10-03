"""Claim-aligned V5 contracts. Typed records enforce consistency, never truth.

Adaptations of factored verification and collaborative debate; no extra agent.
Legacy transports remain readable for controlled baseline comparisons.
"""
from copy import deepcopy
from itertools import permutations
import json

from engine.output_contracts import object_schema, validate_shape, incomplete_clause

ALIGNMENT_VERSION = "aligned-reading-1"
AUDIT_VERSION = "focused-audit-1"
AUDIT_ONLY_VERSION = "audit-only-1"
MODES = {"baseline", "process-audit-1", ALIGNMENT_VERSION, AUDIT_VERSION, AUDIT_ONLY_VERSION}
RULES = (
    "Evaluate the original assertion under the V-FLUTE visual-entailment task. "
    "Keep the asserted property, participants, polarity, modality and qualifiers. "
    "For metaphor, idiom or analogy resolve the depicted referent and transferred property; "
    "physical presence of the comparison's source object is unnecessary. "
    "For sarcastic praise distinguish the expressed evaluation from the speaker's implied complaint: "
    "do not replace praise with the complaint before testing the claim. Sarcasm alone does not determine a label. "
    "A joke or reaction can be supported by a grounded correspondence without literal historical participants. "
    "Test the property using the observations, not just subject or topic overlap. "
    "A reading is a hypothesis, not a visual fact. Missing evidence alone is not contradiction. "
    "Keep uncertainty when a necessary mapping or condition cannot be established. "
)
TEXT = {"type": "string", "minLength": 1}
READING = object_schema({
    "kind": {"type": "string", "enum": ["LITERAL", "FIGURATIVE", "UNCERTAIN"]},
    "referent": TEXT, "property": TEXT,
})


READING_INSTRUCTIONS = (
    "Generate reading first: kind is LITERAL for a direct assertion, FIGURATIVE for a licensed "
    "metaphor, joke, idiom or sarcasm, or UNCERTAIN when no reading can be established. "
    "referent names the pictured entity or reaction being described, not the metaphor's source object. "
    "property states the exact attributed quality, action or relation, retaining negation and qualifiers. "
    "Use brief phrases for referent and property. Then evaluate that property; UNCERTAIN requires "
    "relation UNRESOLVED. The kind alone never decides support or conflict. "
)


def mode(runtime):
    return getattr(runtime, "tribunal_audit_mode", "baseline")


def aligned(runtime):
    return mode(runtime) in {ALIGNMENT_VERSION, AUDIT_VERSION}


def reading_schema(base):
    return object_schema({"reading": deepcopy(READING), **deepcopy(base["properties"])})


def reading_error(value):
    reading = value.get("reading", {})
    if not validate_shape(reading, READING):
        return "Provide the reading kind, depicted referent and asserted property."
    unresolved = {"", "none", "unknown", "unresolved", "uncertain"}
    if value.get("relation") in {"SUPPORT", "CONFLICT"} and (
        reading["kind"] == "UNCERTAIN"
        or any(reading[key].strip().lower() in unresolved for key in ("referent", "property"))
    ):
        return "A directional relation needs an identified referent and property; otherwise use UNRESOLVED."


def audit_schema(known):
    ids = {"type": "array", "minItems": 1, "maxItems": 4,
           "items": {"type": "string", "enum": sorted(known)}}
    return object_schema({
        "alignment": {"type": "string", "enum": ["MATCH", "MISMATCH", "UNRESOLVED"]},
        "objections": {"type": "array", "maxItems": 3, "items": object_schema({
            "type": {"type": "string", "enum": [
                "SCOPE_ERROR", "UNSUPPORTED_INFERENCE", "COUNTEREVIDENCE", "AMBIGUOUS_READING"]},
            "disputed_step": TEXT, "basis": TEXT, "evidence_ids": ids,
        })},
        "evidence_ids": ids,
        "reason": TEXT,
    })


def audit_generation_schema(known):
    """Match the wire's unique-citation requirement in the bounded decoder.

    The grammar backend does not support uniqueItems. Enumerating the legal
    ordered lists is small for the verifier's observation sets; larger sets
    retain the ordinary schema and the same validator and final wire checks.
    """
    schema = audit_schema(known)
    if 0 < len(known) <= 6:
        ids = sorted(known)
        choices = [list(items) for size in range(1, min(4, len(ids)) + 1)
                   for items in permutations(ids, size)]
        schema["properties"]["evidence_ids"]["enum"] = choices
        schema["properties"]["objections"]["items"]["properties"]["evidence_ids"]["enum"] = choices
    return schema


AUDIT_INSTRUCTIONS = (
    "Audit the candidate after the independent source assessment. "
    "Check whether candidate and verifier address the SAME assertion, referent, property, "
    "qualifiers and reading convention. alignment is MATCH only when they do. "
    "Then test the candidate inference against the actual sources. Agreement alone is not proof. "
    "Report only active objections that invalidate that inference or leave material ambiguity. "
    "Each objection names the disputed step and its counterevidence, missing necessary condition "
    "or grounded competing reading. Cite selected observations. "
    "Do not object merely because the caption is false when the candidate correctly identifies conflict. "
    "Do not invent an alternative or retain an objection your own reason resolves. "
    "Use an empty objections array when none remains. An incomplete or contradictory audit "
    "cannot authorize acceptance. Give one complete sentence for reason; keep other text brief. "
)


def audit_error(value):
    citations = [value["evidence_ids"]] + [item["evidence_ids"] for item in value["objections"]]
    if any(len(ids) != len(set(ids)) for ids in citations):
        return {"kind": "AUDIT_DUPLICATE_CITATION",
                "message": "Cite each selected observation at most once within each evidence_ids list."}
    if value["alignment"] != "MATCH" and not value["objections"]:
        return {"kind": "AUDIT_INCOMPLETE",
                "message": "Name the material mismatch or uncertainty in an objection."}
    for item in value["objections"]:
        if not item["disputed_step"].strip() or not item["basis"].strip():
            return {"kind": "AUDIT_INCOMPLETE", "message": "An objection needs a disputed step and a basis."}
    if incomplete_clause(value["reason"], conjunctions=True):
        return {"kind": "AUDIT_INCOMPLETE", "message": "Finish the audit reason."}


def project_audit(call):
    """Single authoritative objection list; legacy fields are exact projections."""
    value = {key: deepcopy(call.get(key)) for key in
             ("alignment", "objections", "evidence_ids", "reason")}
    objections = value.get("objections") or []
    errors = [f"{item['disputed_step']}: {item['basis']}" for item in objections
              if item["type"] != "SCOPE_ERROR"]
    roles = [f"{item['disputed_step']}: {item['basis']}" for item in objections
             if item["type"] == "SCOPE_ERROR"]
    alternative = next((x for x in objections if x["type"] == "AMBIGUOUS_READING"), None)
    call.update(
        _focused_audit_protocol=AUDIT_VERSION, _focused_value=value,
        decision_errors=errors, role_scope_errors=roles,
        alternative=alternative["disputed_step"] if alternative else "",
        alternative_relation="UNRESOLVED",
        alternative_status="UNRESOLVED" if alternative else "NONE",
        deciding_evidence_ids=list(value.get("evidence_ids") or []),
    )
    return call


def audit_wire_complete(call, known):
    """A delivered record may still be inconsistent; this never grants acceptance."""
    if call.get("_focused_audit_protocol") != AUDIT_VERSION:
        return False
    try:
        value = json.loads(call.get("_raw_output", ""))
    except (ValueError, TypeError):
        return False
    if not validate_shape(value, audit_schema(known)):
        return False
    if value != call.get("_focused_value"):
        return False
    projected = project_audit(deepcopy(value))
    keys = ("alignment", "objections", "evidence_ids", "reason", "decision_errors",
            "role_scope_errors", "alternative", "alternative_relation",
            "alternative_status", "deciding_evidence_ids")
    if any(call.get(key) != projected.get(key) for key in keys):
        return False
    return all(len(x) == len(set(x)) for x in
               [value["evidence_ids"]] + [o["evidence_ids"] for o in value["objections"]])


def audit_response_valid(call, known):
    """Require successful execution, exact provenance and a consistent audit."""
    from engine.evidence_verification import executed
    return (executed(call) and audit_wire_complete(call, known)
            and not audit_error(call["_focused_value"]))


def audit_accepts(call, known):
    return (audit_response_valid(call, known) and call["alignment"] == "MATCH"
            and not call["objections"])


def dispute_key(review, issue, diagnostic):
    """Novelty is tied to the failed premise and sources, not a question paraphrase."""
    import hashlib
    packet = review.get("_judge_packet") or {}
    proof = review.get("_independent_verification") or {}
    data = [issue, diagnostic, packet.get("source_sha256"),
            packet.get("observations"), proof.get("selected_evidence_ids")]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
