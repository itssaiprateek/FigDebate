"""Independent, label-blind Qwen review of the completed FigDebate case."""

import json
import inspect
import time

from models.judge_model import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from utils.judge_parser import (
    parse_judge_response,
    parse_mediation_response,
    parse_tribunal_review_response,
)


JUDGE_SCHEMA_VERSION = "1.0"


def _text(value, limit=320):
    return str(value or "").strip()[:limit]


def _list(value, limit=5):
    return [_text(item) for item in (value or [])[:limit] if _text(item)]


def _visual_packet(output):
    output = output or {}
    return {
        "visual_facts": _list(output.get("visual_facts")),
        "visible_text": _list(output.get("visible_text")),
        "visual_relations": _list(output.get("visual_relations")),
        "visual_metaphors": _list(
            output.get("possible_visual_metaphors", output.get("visual_metaphors"))
        ),
        "symbolic_tone": _text(output.get("symbolic_tone")),
        "confidence": output.get("confidence"),
    }


def _language_packet(output):
    output = output or {}
    relation = output.get("claim_relation", {}) or {}
    contract = output.get("claim_contract", {}) or {}
    return {
        "caption_proposition": _text(output.get("caption_proposition")),
        "intended_meaning": _text(output.get("intended_meaning")),
        "figurative_type": _text(output.get("figurative_type")),
        "linguistic_cue": _text(output.get("linguistic_cue")),
        "polarity_reversal": output.get("polarity_reversal"),
        "relation": {
            "family": relation.get("relation_family"),
            "polarity": relation.get("polarity"),
            "predicate": relation.get("predicate"),
            "resolved": relation.get("resolved"),
        },
        "claim_contract": {
            "safe_for_directional_reasoning": contract.get(
                "safe_for_directional_reasoning"
            ),
            "safe_for_tribunal_reasoning": contract.get(
                "safe_for_tribunal_reasoning"
            ),
            "warnings": _list(contract.get("warnings")),
            "structural_reasoning_type": contract.get(
                "structural_reasoning_type"
            ),
            "figurative_mechanism_candidates": _list(
                contract.get("figurative_mechanism_candidates")
            ),
            "literal_polarity": contract.get("literal_polarity"),
            "intended_polarity": contract.get("intended_polarity"),
            "comparison_direction": contract.get("comparison_direction"),
            "evaluation_target": contract.get("evaluation_target"),
            "time_or_panel_scope": contract.get("time_or_panel_scope"),
        },
    }


def _comparison_packet(output):
    output = output or {}
    return {
        "recommendation": _text(output.get("recommendation")),
        "required_evidence_status": _text(output.get("required_evidence_status")),
        "supporting_evidence": _list(output.get("supporting_evidence")),
        "contradicting_evidence": _list(output.get("contradicting_evidence")),
        "missing_evidence": _list(output.get("missing_evidence")),
        "evidence_quality": output.get("evidence_quality"),
        "relation_binding_required": output.get("relation_binding_required"),
        "relation_binding_observed": output.get("relation_binding_observed"),
        "claim_direction": output.get("claim_direction"),
        "direct_support_count": output.get("direct_support_count", 0),
        "direct_conflict_count": output.get("direct_conflict_count", 0),
        "structured_observations": list(
            output.get("structured_observations", []) or []
        )[:12],
    }


def _critique_packet(debate):
    debate = debate or {}
    packets = []
    for role, key in (
        ("visual_reviewer", "agent1_critique"),
        ("claim_reviewer", "agent2_critique"),
    ):
        critique = debate.get(key, {}) or {}
        if critique:
            packets.append({
                "role": role,
                "stance": _text(critique.get("stance")),
                "recommendation": _text(critique.get("recommendation")),
                "reason": _text(critique.get("reason"), 800),
                "observed_entity": _text(critique.get("observed_entity")),
                "observed_state": _text(critique.get("observed_state")),
                "image_region": _text(critique.get("image_region")),
                "specific_evidence": critique.get("specific_evidence"),
                "response_status": _text(critique.get("response_status")),
                "witness_contract": critique.get("witness_contract", {}),
                "support_requirement": _text(
                    critique.get("support_requirement"), 600
                ),
                "conflict_requirement": _text(
                    critique.get("conflict_requirement"), 600
                ),
                "requirements_valid": critique.get("requirements_valid"),
            })
    return packets


def _ledger_packet(ledger, limit=18):
    values = sorted(
        list(ledger or []),
        key=lambda item: (
            not bool(
                item.get("decision_grade", False)
                or item.get("verification", {}).get("decision_grade", False)
            ),
            not bool(item.get("grounded", False)),
            str(item.get("id", "")),
        ),
    )
    if limit is not None:
        values = values[:limit]
    return [
        {
            "id": item.get("id"),
            "source": item.get("source"),
            "type": item.get("type"),
            "text": _text(item.get("text"), 320),
            "relation": item.get("relation"),
            "grounded": bool(item.get("grounded", False)),
            "decision_grade": bool(
                item.get("decision_grade", False)
                or item.get("verification", {}).get("decision_grade", False)
            ),
        }
        for item in values
    ]


def build_judge_packet(
    caption,
    visual_output,
    language_output,
    comparison,
    evidence_ledger,
    debate_details=None,
):
    """Build a compact packet without the primary Arbiter label or gold label."""
    return {
        "caption": _text(caption, 1000),
        "visual_agent": _visual_packet(visual_output),
        "claim_agent": _language_packet(language_output),
        "deterministic_comparator": _comparison_packet(comparison),
        "debate_reviews": _critique_packet(debate_details),
        "evidence_ledger": _ledger_packet(evidence_ledger, limit=None),
    }


def build_mediation_packet(
    caption,
    visual_output,
    language_output,
    comparison,
    evidence_ledger,
):
    """Expose the complete current ledger without any Arbiter or gold label."""
    packet = build_judge_packet(
        caption,
        visual_output,
        language_output,
        comparison,
        evidence_ledger,
        debate_details=None,
    )
    packet["evidence_ledger"] = _ledger_packet(evidence_ledger, limit=None)
    packet.pop("debate_reviews", None)
    return packet


def build_judge_prompt(packet):
    return """Review the original image and caption independently, then audit the two agents,
the comparator, and both debate reviewers. The current Arbiter decision and the gold label
are intentionally hidden. Do not infer them.

Decision meanings:
- ENTAILS: the image provides direct support for the caption's intended claim.
- CONTRADICTS: the image provides direct evidence that conflicts with that intended claim.
- ABSTAIN: the supplied evidence is ambiguous, missing, or only shows lack of support.

Rules:
1. Missing support alone is not contradiction.
2. Treat agent text as claims to audit, not as visual fact.
3. Cite only IDs present in evidence_ledger.
4. A decisive verdict should cite decision_grade evidence with the matching relation.
5. Return exactly one JSON object and no Markdown or extra fields:
{"verdict":"ENTAILS|CONTRADICTS|ABSTAIN","confidence":0.0,"evidence_ids":["ID"],"visual_observations":["direct observation"],"reason":"short audit"}

CASE PACKET:
""" + json.dumps(packet, ensure_ascii=True, sort_keys=True)


def build_mediation_prompt(packet):
    return """Act as a label-blind mediator before the two-agent debate. Inspect the
original image and caption, then audit the visual agent, claim agent, comparator,
and every current evidence-ledger entry. The current Arbiter label and gold label
are intentionally hidden.

Your job is to improve the debate, not to decide it by authority:
1. Identify only disagreements that can be resolved from the current image.
2. Ask the visual reviewer to reinspect a precise entity, state, text binding, or region.
3. Ask the claim reviewer to clarify the exact proposition, polarity, or figurative relation.
4. Cite only IDs present in evidence_ledger. Unknown or invented IDs invalidate the plan.
5. Treat your own observations and questions as advisory, never as verified evidence.
6. Use ABSTAIN when no targeted check can safely resolve the issue.
7. Be terse: each string must be at most 20 words. For ABSTAIN, question and
   verification strings may be empty, but issue must explain why.
8. Return exactly one JSON object and no Markdown or extra fields:
{"status":"MEDIATE|ABSTAIN","provisional_verdict":"ENTAILS|CONTRADICTS|ABSTAIN","confidence":0.0,"evidence_ids":["ID"],"issue":"one issue","agent1_question":"one visual question","agent2_question":"one claim question","verification_request":"one check"}

CASE PACKET:
""" + json.dumps(packet, ensure_ascii=True, sort_keys=True)


def build_tribunal_review_prompt(packet, round_number):
    return f"""You are the label-blind mediator reviewing tribunal round {round_number}.
You can inspect the original image, caption, agent answers, comparator candidates,
and complete evidence ledger. The gold label is hidden.

Decide whether the dispute is resolved, requires one targeted follow-up, or must
remain unresolved. Missing evidence is never contradiction. Lexical relation
candidates are not verified evidence. Cite only supplied ledger IDs. Your own
visual observations are advisory and can never become decision-grade proof by
themselves.

Rules:
1. RESOLVE requires SUPPORT or CONFLICT, at least one cited current-round
   visual witness, and one direct observation. The deterministic resolver will
   decide whether independent corroboration is sufficient.
2. FOLLOW_UP requires UNRESOLVED plus one neutral atomic question for either agent.
3. ABSTAIN requires UNRESOLVED and no leading questions.
4. Ask about observable facts for Agent 1 and caption meaning for Agent 2.
5. Keep every string under 16 words and visual_observations to at most two.
6. Relation means the relation between the cited observation and the preserved
   caption proposition: SUPPORT, CONFLICT, or UNRESOLVED. Never output a dataset label.
7. In round 2, FOLLOW_UP is forbidden; choose RESOLVE or ABSTAIN.
8. First resolve the structural evidence type (OCR binding, comparison,
   event order, reaction target, affect, or symbol attachment), then apply the
   figurative mechanism. Do not let "humor" erase criticism or polarity.
9. For metaphor, keep literal source observations separate from the caption
   target and verify the transferred property explicitly.
10. For sarcasm, keep literal wording, intended polarity, evaluation target,
    and visible referent separate.
11. For multi-panel scenes, verify actor, action, immediate effect, later
    outcome, and blamed target in order. Temporal order alone is not causation.
12. A CONFLICT resolution requires an affirmative observed opposite, never a
    missing expected feature. A SUPPORT resolution requires an affirmative
    observed match.
13. Return exactly one JSON object and no extra fields:
{{"status":"RESOLVE|FOLLOW_UP|ABSTAIN","relation":"SUPPORT|CONFLICT|UNRESOLVED","confidence":0.0,"evidence_ids":["ID"],"visual_observations":["direct observation"],"issue":"unresolved issue","agent1_question":"question or empty","agent2_question":"question or empty","verification_request":"verification or empty","reason":"short evidence audit"}}

CASE PACKET:
""" + json.dumps(packet, ensure_ascii=True, sort_keys=True)


def _run_structured_generation(
    runtime, image, prompt, parser, *, max_new_tokens, contract_name
):
    """Generate, validate, and perform one bounded format-repair retry."""
    total_seconds = 0.0
    attempts = []
    last_output = ""
    parsed = parser("")
    try:
        for attempt in range(2):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nFORMAT REPAIR: The previous response violated the JSON "
                    f"contract ({parsed.get('_format_error', 'invalid output')}). "
                    "Return one compact JSON object only. Do not repeat text."
                )
            try:
                parameters = inspect.signature(runtime.generate).parameters.values()
                supports_token_limit = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    or parameter.name == "max_new_tokens"
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                supports_token_limit = True
            if supports_token_limit:
                generated = runtime.generate(
                    image, attempt_prompt, max_new_tokens=max_new_tokens
                )
            else:
                # Preserve compatibility with injected test/research runtimes
                # that implement the original two-argument contract.
                generated = runtime.generate(image, attempt_prompt)
            if isinstance(generated, tuple):
                last_output, elapsed = generated
            else:
                last_output = generated
                elapsed = 0.0
            total_seconds += float(elapsed)
            diagnostics = dict(
                getattr(runtime, "_last_generation_diagnostics", {}) or {}
            )
            attempts.append(diagnostics)
            parsed = parser(last_output)
            if (
                parsed.get("_format_valid", False)
                and not diagnostics.get("hit_token_limit", False)
            ):
                break
    except Exception as error:
        parsed = parser("")
        parsed["_format_error"] = f"{contract_name}_generation_failed:{error}"
    parsed["_format_retry_used"] = len(attempts) > 1
    parsed["_format_retry_success"] = bool(
        len(attempts) > 1 and parsed.get("_format_valid", False)
    )
    parsed["_generation_diagnostics"] = attempts
    parsed["_generation_seconds"] = round(total_seconds, 4)
    if not parsed.get("_raw_output"):
        parsed["_raw_output"] = str(last_output or "")
    return parsed


class MultimodalJudgeAgent:
    def __init__(self, runtime):
        self.runtime = runtime

    def analyze(
        self,
        image,
        caption,
        visual_output,
        language_output,
        comparison,
        evidence_ledger,
        debate_details=None,
    ):
        packet = build_judge_packet(
            caption,
            visual_output,
            language_output,
            comparison,
            evidence_ledger,
            debate_details,
        )
        prompt = build_judge_prompt(packet)
        judgment = _run_structured_generation(
            self.runtime,
            image,
            prompt,
            parse_judge_response,
            max_new_tokens=256,
            contract_name="judge",
        )

        known_ids = {item.get("id") for item in (evidence_ledger or [])}
        cited = judgment.get("evidence_ids", [])
        judgment["_valid_evidence_ids"] = [
            item_id for item_id in cited if item_id in known_ids
        ]
        judgment["_invalid_evidence_ids"] = [
            item_id for item_id in cited if item_id not in known_ids
        ]
        judgment["_model_id"] = JUDGE_MODEL_ID
        judgment["_model_revision"] = JUDGE_MODEL_REVISION
        judgment["_schema_version"] = JUDGE_SCHEMA_VERSION
        return judgment


class MultimodalMediatorAgent:
    """Create targeted debate questions without changing evidence or labels."""

    def __init__(self, runtime):
        self.runtime = runtime

    def analyze(
        self,
        image,
        caption,
        visual_output,
        language_output,
        comparison,
        evidence_ledger,
    ):
        packet = build_mediation_packet(
            caption,
            visual_output,
            language_output,
            comparison,
            evidence_ledger,
        )
        prompt = build_mediation_prompt(packet)
        mediation = _run_structured_generation(
            self.runtime,
            image,
            prompt,
            parse_mediation_response,
            max_new_tokens=256,
            contract_name="mediator",
        )

        known_ids = {item.get("id") for item in (evidence_ledger or [])}
        cited = mediation.get("evidence_ids", [])
        mediation["_valid_evidence_ids"] = [
            item_id for item_id in cited if item_id in known_ids
        ]
        mediation["_invalid_evidence_ids"] = [
            item_id for item_id in cited if item_id not in known_ids
        ]
        mediation["_usable"] = bool(
            mediation.get("_format_valid", False)
            and mediation.get("status") == "MEDIATE"
            and not mediation["_invalid_evidence_ids"]
        )
        mediation["_model_id"] = JUDGE_MODEL_ID
        mediation["_model_revision"] = JUDGE_MODEL_REVISION
        mediation["_schema_version"] = JUDGE_SCHEMA_VERSION
        return mediation


class TribunalMediatorAgent:
    """Review agent answers and request at most one additional tribunal round."""

    def __init__(self, runtime):
        self.runtime = runtime

    def review(
        self, image, caption, visual_output, language_output, comparison,
        evidence_ledger, debate_details, round_number=1,
    ):
        packet = build_judge_packet(
            caption, visual_output, language_output, comparison,
            evidence_ledger, debate_details,
        )
        packet["tribunal_round"] = int(round_number)
        prompt = build_tribunal_review_prompt(packet, round_number)
        review = _run_structured_generation(
            self.runtime,
            image,
            prompt,
            parse_tribunal_review_response,
            max_new_tokens=320,
            contract_name="tribunal_review",
        )
        if int(round_number) >= 2 and review.get("status") == "FOLLOW_UP":
            review = dict(review)
            review.update({
                "status": "ABSTAIN",
                "relation": "UNRESOLVED",
                "provisional_verdict": "ABSTAIN",
                "agent1_questions": [],
                "agent2_questions": [],
                "verification_requests": [],
                "reason": (
                    review.get("reason") or "Maximum tribunal rounds reached."
                ),
                "_terminal_normalization": "maximum_rounds_reached",
            })

        known_ids = {item.get("id") for item in (evidence_ledger or [])}
        review["_valid_evidence_ids"] = [
            item_id for item_id in review.get("evidence_ids", [])
            if item_id in known_ids
        ]
        review["_invalid_evidence_ids"] = [
            item_id for item_id in review.get("evidence_ids", [])
            if item_id not in known_ids
        ]
        review["_model_id"] = JUDGE_MODEL_ID
        review["_model_revision"] = JUDGE_MODEL_REVISION
        review["_schema_version"] = "tribunal-1.0"
        return review
