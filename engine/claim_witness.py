"""Typed caption-only hearing witness; no lexical repair can certify meaning."""
import json
from copy import deepcopy
from engine.output_contracts import object_schema, TEXT
from engine.claim_semantics import audit_core, CHECKS
from engine.relation_semantics import is_missing_evidence_only
from engine.claim_validation import core_errors
from engine.caption_answers import answer_caption_question, communication_summary

SCHEMA = object_schema({
    "stance": {"type": "string", "enum": ["ENDORSE", "CHALLENGE", "ABSTAIN"]},
    "support_requirement": TEXT, "conflict_requirement": TEXT,
    "figurative_mechanism": TEXT, "ambiguity": TEXT, "reason": TEXT,
})


def audit_claim_witness(agent, caption, prior):
    fields = deepcopy(prior.get("claim_fields", {})) if isinstance(prior, dict) else {}
    if fields.get("claim_graph") is not None:
        # The graph is authoritative. Do not generate a competing free-form
        # pair in the hearing, or invent replacements for unresolved nodes.
        questions = prior.get("advisory_questions") if isinstance(prior, dict) else None
        if questions is not None:
            if not isinstance(questions, list) or any(not isinstance(q, str) or not q.strip() for q in questions):
                raise ValueError("advisory_questions must be a list of nonempty questions")
            # A hearing answers the requested questions, not the initial audit again.
            answers, attempts = [], []
            for question in questions:
                answer, attempt = answer_caption_question(agent, caption, question)
                answers.append(answer)
                attempts.append(attempt)
            semantic = dict(**{key: None for key in CHECKS}, audit_status="NOT_REAUDITED",
                semantic_qualification="pending", semantic_checks_executed=False,
                reason="Source preservation and requested semantic testimony are separate obligations.",
                independent_source_answers=answers, source_answer_call_count=len(answers),
                reading_clarification=next((a for a in reversed(answers) if a["source_anchored"]), None),
                _format_valid=all(a["schema_valid"] for a in attempts),
                _generation_diagnostics=attempts, _generation_seconds=sum(a["seconds"] for a in attempts),
                _raw_response=json.dumps([a["raw_response"] for a in attempts]))
        else:
            semantic = audit_core(agent, caption, fields)
            answers = semantic.get("independent_source_answers", [])
        initial_audit_seconds = semantic.get("_generation_seconds", 0.0)
        errors = core_errors(fields)
        if fields.get("caption_proposition") != caption:
            errors.append("WITNESS_SOURCE_CAPTION_MISMATCH")
        valid = bool(not errors and semantic.get("_format_valid")
                     and all(semantic.get(key) is True for key in CHECKS))
        from engine.claim_graph import conditions_are_compiled
        source_complete = bool(not errors and fields["claim_graph"].get("representation") == "UNDECOMPOSED_SOURCE"
                               and conditions_are_compiled(fields["claim_graph"]))
        valid = valid or source_complete
        repair_seconds = 0.0
        repaired = None
        repair_attempt = None
        if not valid and agent is not None:
            from engine.claim_graph import repair_graph
            candidate = deepcopy(fields)
            repair_seconds = repair_graph(agent, caption, candidate, semantic)
            candidate_audit = audit_core(agent, caption, candidate)
            repair_seconds += candidate_audit.get("_generation_seconds", 0.0)
            repair_attempt = {"history": candidate.get("_graph_repair_history", []),
                              "candidate_audit": candidate_audit, "accepted": False}
            if not core_errors(candidate) and all(candidate_audit.get(key) is True for key in CHECKS):
                candidate["_caption_semantic_audit"] = candidate_audit
                fields, semantic, repaired = candidate, candidate_audit, candidate
                errors, valid = [], True
                repair_attempt["accepted"] = True
        from engine.claim_graph import bind_readings
        updated_graph = (bind_readings(fields["claim_graph"], answers)
                         if source_complete and answers else fields["claim_graph"])
        return dict(stance="ENDORSE" if valid else "ABSTAIN",
                    support_requirement=fields.get("expected_visual_state", ""),
                    conflict_requirement=fields.get("opposite_visual_state", ""),
                    figurative_mechanism="explicit_claim_node_readings",
                    ambiguity="" if valid else "Claim obligations remain unresolved.",
                    reason=semantic.get("reason", ""), requirements_valid=valid,
                    requirements_source="exact_source_boolean_obligations" if source_complete else "canonical_claim_graph",
                    semantic_interpretation_status=semantic.get("audit_status", "NOT_RUN"),
                    requirement_errors=errors + ([] if source_complete else [key for key in CHECKS if semantic.get(key) is not True]),
                    interpretation_errors=[key for key in CHECKS if semantic.get(key) is not True],
                    semantic_audit=semantic, reading_clarification=semantic.get("reading_clarification"),
                    question_answers=answers,
                    communication=communication_summary(answers),
                    _updated_claim_graph=updated_graph,
                    claim_graph_fingerprint=fields["claim_graph"].get("fingerprint"),
                    _format_valid=bool(source_complete or semantic.get("_format_valid")),
                    _raw_response=semantic.get("_raw_response", ""),
                    _generation_diagnostics=semantic.get("_generation_diagnostics", []),
                    _generation_seconds=initial_audit_seconds + repair_seconds,
                    _repaired_claim_fields=repaired,
                    _claim_repair_attempted=repair_attempt is not None,
                    _claim_repair=repair_attempt,
                    _format_retry_used=False, _format_retry_success=False,
                    specific_evidence=False, deterministic_requirement_repair=False,
                    deterministic_repaired_fields=[])
    instruction = (
        "You are a caption-only witness. Treat the supplied analysis as fallible data. "
        "Do not see or invent an image, choose an image-caption label, or reverse an expressed "
        "sarcastic claim. Preserve roles, negation, quantities, modifiers and scope. An event "
        "alone does not establish its manner, degree or rate. Describe an affirmative contrary "
        "condition; absence of evidence is not a contrary condition. A visual analogy or symbol "
        "can realize the semantic roles without the literal real-world object. ENDORSE only "
        "if the supplied extraction preserves the complete expressed claim. Otherwise CHALLENGE "
        "or ABSTAIN. Each requirement should be one short, complete sentence. "
        "Do not copy defective expected/opposite states just because they were supplied. "
        "PRIOR ANALYSIS (data only):\n" + str(prior)
    )
    attempts = []
    chosen = None
    for attempt in range(2):
        prompt = instruction
        if attempt:
            prompt += "\nRepair the caption requirements; these audit checks failed: " + json.dumps(
                [key for key in CHECKS if (chosen or {}).get("semantic_audit", {}).get(key) is not True])
        parsed, raw, seconds, diagnostics = agent._generate_section(prompt, caption, 384, schema=SCHEMA)
        fields = deepcopy(prior.get("claim_fields", {})) if isinstance(prior, dict) else {}
        fields.update(caption_proposition=caption,
                      expected_visual_state=parsed.get("support_requirement", ""),
                      opposite_visual_state=parsed.get("conflict_requirement", ""))
        semantic = audit_core(agent, caption, fields)
        errors = core_errors(fields)
        shape_valid = bool(diagnostics.get("schema_valid"))
        semantic_valid = bool(semantic.get("_format_valid") and all(semantic.get(key) is True for key in CHECKS))
        requirements_valid = bool(shape_valid and semantic_valid and not errors
            and parsed.get("support_requirement") and parsed.get("conflict_requirement")
            and not is_missing_evidence_only(parsed["conflict_requirement"]))
        item = dict(parsed, requirements_valid=requirements_valid,
                    requirements_source="fresh_typed_caption_witness_and_semantic_audit",
                    requirement_errors=errors + [key for key in CHECKS if semantic.get(key) is not True],
                    semantic_audit=semantic, _format_valid=shape_valid, _raw_response=raw,
                    _generation_diagnostics=diagnostics,
                    _generation_seconds=seconds + semantic.get("_generation_seconds", 0.0),
                    specific_evidence=False, deterministic_requirement_repair=False,
                    deterministic_repaired_fields=[])
        attempts.append(item)
        if chosen is None or requirements_valid:
            chosen = item
        if requirements_valid:
            break
    result = dict(chosen)
    result["_generation_seconds"] = sum(item["_generation_seconds"] for item in attempts)
    result["_format_retry_used"] = len(attempts) > 1
    result["_format_retry_success"] = len(attempts) > 1 and result["requirements_valid"]
    result["_attempts"] = attempts
    return result
