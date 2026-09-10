"""Factored caption verification. Source answers never see the proposal.

Adapted from Chain-of-Verification (arXiv:2309.11495); this smaller-model
implementation still requires semantic qualification, not just JSON tests.
"""
import json
from engine.output_contracts import CORE, TEXT, TEXT_LIST, object_schema, validate_shape
from engine.claim_validation import core_errors
from engine.caption_answers import answer_caption_question

CHECKS = ("expressed_claim_preserved", "entity_roles_preserved",
          "negation_quantities_preserved", "modifiers_scope_preserved",
          "expected_state_supports_full_claim", "opposite_state_conflicts_with_full_claim")
# Retained solely for replaying the archived Boolean-audit diagnostic.
SCHEMA = object_schema({
    **{key: {"type": "boolean"} for key in CHECKS},
    "defective_fields": {"type": "array", "items": {"type": "string", "enum": list(CORE["properties"])}},
    "reason": TEXT,
})
ANSWER_SCHEMA = object_schema({"answer": TEXT, "source_quotes": TEXT_LIST, "unknown": {"type": "boolean"}})
COMPARISON_SCHEMA = object_schema({
    "status": {"type": "string", "enum": ["PASS", "FAIL", "UNKNOWN"]},
    "reason": TEXT,
})
RELATION_SCHEMA = object_schema({
    "relation": {"type": "string", "enum": ["SUPPORT", "CONFLICT", "UNRESOLVED"]},
    "reason": TEXT,
})
SOURCE_QUESTIONS = (
    "What exactly does the caption assert, about whom or what? Explain who does what to whom, "
    "including every negation, quantity, comparison, degree, manner and time/panel qualifier. "
    "Quote the words establishing those details.",
    "Which readings does the wording permit? Explain any idiom without requiring its literal "
    "physical event. Distinguish the expressed proposition from a sarcastic intended message. "
    "What would establish the full proposition, and what affirmative contrary condition would "
    "refute a necessary part? Do not treat missing evidence as refutation. Quote the relevant words.",
)
GROUPS = (
    (CHECKS[:4], [key for key in CORE["properties"] if key not in
                  ("expected_visual_state", "opposite_visual_state")],
     "Does this proposed representation preserve the full expressed proposition, roles, "
     "negation, quantities and all modifiers/scope in the independent source answer?"),
    ((CHECKS[4],), ["expected_visual_state"],
     "Would this proposed condition establish the entire expressed claim under the justified "
     "reading? Establishing just the event without its qualifier is insufficient."),
    ((CHECKS[5],), ["opposite_visual_state"],
     "Would this proposed condition affirmatively conflict with a necessary part of the "
     "expressed claim? It need not reverse every conjunct. Absence alone is not conflict."),
)


def audit_core(agent, caption, fields):
    subject = {key: fields.get(key) for key in CORE["properties"]}
    errors = core_errors(fields)
    # None means not evaluated, never a failed semantic comparison.
    result = {key: None for key in CHECKS}
    attempts, answers, obligations = [], [], []
    seconds = 0.0
    model_comparison_count = 0

    def generate(instruction, schema, budget):
        nonlocal seconds
        value, raw, elapsed, diagnostics = agent._generate_section(instruction, caption, budget, schema=schema)
        seconds += elapsed
        valid = bool(diagnostics.get("schema_valid")) and validate_shape(value, schema)
        attempts.append(dict(prompt=instruction, raw_response=raw, schema=schema,
                             diagnostics=diagnostics, schema_valid=valid))
        return value if valid else {}, valid

    # Invalid input never gains authority from an always-approving model.
    if not errors:
        for question in SOURCE_QUESTIONS:
            answer, attempt = answer_caption_question(agent, caption, question)
            seconds += attempt["seconds"]
            attempts.append(attempt)
            answers.append(answer)
        for index, (checks, keys, question) in enumerate(GROUPS):
            from engine.claim_graph import conditions_are_compiled
            compiled = conditions_are_compiled(fields.get("claim_graph") or {})
            if index and compiled:
                # Sound only relative to a source-checked graph: conjunction/
                # disjunction projection implements P / not-P exactly. This
                # establishes a hypothetical obligation, never image truth.
                source_passed = all(result[key] is True for key in CHECKS[:4])
                comparison = {"status": "PASS" if source_passed else "UNKNOWN",
                    "reason": "Canonical truth-functional condition over source-checked propositions."
                              if source_passed else "Source graph meaning not established.",
                    "method": "canonical_boolean_projection_not_model_nli"}
                obligations.append(dict(comparison, checks=list(checks), fields=keys))
                for key in checks:
                    result[key] = True if source_passed else None
                continue
            # The reading/idiom answer is a dependency, not unused decoration.
            relevant = answers
            if not all(item.get("source_anchored") and not item.get("unknown") for item in relevant):
                comparison = {"status": "UNKNOWN", "reason": "Independent source answer unavailable or unanchored."}
            else:
                instruction = (
                    "Compare only this obligation. PASS requires the complete obligation; "
                    "FAIL requires an identified inconsistency; use UNKNOWN when not established. "
                    "Neither the proposal nor the independent answer is guaranteed correct. "
                    + question + "\nIndependent source answers: " + json.dumps(relevant)
                    + "\nProposed fields (fallible data): " + json.dumps({key: subject[key] for key in keys}))
                schema = COMPARISON_SCHEMA
                if index == 0 and fields.get("claim_graph") is not None:
                    instruction += "\nAlso verify that these atoms cover the entire caption, with correct bindings, "
                    instruction += "logical composition and idiomatic mappings: " + json.dumps(fields["claim_graph"])
                if index:
                    # Same neutral relation task for both conditions: no target
                    # field name or desired answer in the model's context.
                    instruction = (
                        "Assume the condition below holds. Determine its relation to the original caption's "
                        "expressed proposition: SUPPORT if it establishes the full claim, CONFLICT if it "
                        "contradicts a necessary part, UNRESOLVED otherwise. This is a hypothetical language "
                        "comparison, not an image observation. Preserve roles, qualifiers and figurative readings. "
                        "Missing evidence does not imply conflict.\nIndependent source reading: "
                        + json.dumps(relevant) + "\nCondition: " + json.dumps(subject[keys[0]]))
                    schema = RELATION_SCHEMA
                comparison, valid = generate(instruction, schema, 192)
                model_comparison_count += 1
                if index and valid:
                    expected = "SUPPORT" if index == 1 else "CONFLICT"
                    relation = comparison["relation"]
                    comparison["status"] = ("UNKNOWN" if relation == "UNRESOLVED" else
                                            "PASS" if relation == expected else "FAIL")
                if not valid or not comparison.get("reason", "").strip():
                    comparison = {"status": "UNKNOWN", "reason": "Invalid comparison output."}
            obligations.append(dict(comparison, checks=list(checks), fields=keys))
            for key in checks:
                result[key] = None if comparison["status"] == "UNKNOWN" else comparison["status"] == "PASS"
    defective = list(CORE["properties"]) if errors else list(dict.fromkeys(
        field for item in obligations if item["status"] != "PASS" for field in item["fields"]))
    return dict(result, audit_status=("BLOCKED_UPSTREAM" if errors else
                "PASS" if all(result.values()) else "FAIL" if any(
                    item["status"] == "FAIL" for item in obligations) else "UNKNOWN"),
                semantic_checks_executed=bool(model_comparison_count),
                model_comparison_count=model_comparison_count,
                source_answer_call_count=len(answers),
                deterministic_projection_count=sum(item.get("method") == "canonical_boolean_projection_not_model_nli" for item in obligations),
                reading_clarification=answers[1] if len(answers) > 1 else None,
                defective_fields=defective, input_errors=errors,
                claim_graph_fingerprint=(fields.get("claim_graph") or {}).get("fingerprint"),
                reason="; ".join(errors or [item["reason"] for item in obligations]),
                obligations=obligations, independent_source_answers=answers,
                _format_valid=not errors and all(item["schema_valid"] for item in attempts),
                _raw_response=json.dumps([item["raw_response"] for item in attempts]),
                _generation_seconds=seconds, _generation_diagnostics=attempts,
                method="factored_source_questions_then_neutral_relation_v2",
                semantic_qualification="pending", model_error_independence_established=False)
