"""Independent, label-blind Qwen review of the completed FigDebate case."""

import json
import time

from models.judge_model import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from utils.judge_parser import parse_judge_response, parse_mediation_response


JUDGE_SCHEMA_VERSION = "1.0"


def _text(value, limit=500):
    return str(value or "").strip()[:limit]


def _list(value, limit=8):
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
            "warnings": _list(contract.get("warnings")),
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
            })
    return packets


def _ledger_packet(ledger, limit=40):
    values = list(ledger or [])
    if limit is not None:
        values = values[:limit]
    return [
        {
            "id": item.get("id"),
            "source": item.get("source"),
            "type": item.get("type"),
            "text": _text(item.get("text"), 600),
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
        "evidence_ledger": _ledger_packet(evidence_ledger),
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
        started = time.time()
        try:
            generated = self.runtime.generate(image, prompt)
            if isinstance(generated, tuple):
                raw_output, generation_seconds = generated
            else:
                raw_output = generated
                generation_seconds = time.time() - started
            judgment = parse_judge_response(raw_output)
        except Exception as error:
            judgment = parse_judge_response("")
            judgment["_format_error"] = f"judge_generation_failed:{error}"
            generation_seconds = time.time() - started

        known_ids = {item.get("id") for item in (evidence_ledger or [])}
        cited = judgment.get("evidence_ids", [])
        judgment["_valid_evidence_ids"] = [
            item_id for item_id in cited if item_id in known_ids
        ]
        judgment["_invalid_evidence_ids"] = [
            item_id for item_id in cited if item_id not in known_ids
        ]
        judgment["_generation_seconds"] = round(generation_seconds, 4)
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
        started = time.time()
        try:
            generated = self.runtime.generate(image, prompt, max_new_tokens=256)
            if isinstance(generated, tuple):
                raw_output, generation_seconds = generated
            else:
                raw_output = generated
                generation_seconds = time.time() - started
            mediation = parse_mediation_response(raw_output)
        except Exception as error:
            mediation = parse_mediation_response("")
            mediation["_format_error"] = f"mediator_generation_failed:{error}"
            generation_seconds = time.time() - started

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
        mediation["_generation_seconds"] = round(generation_seconds, 4)
        mediation["_model_id"] = JUDGE_MODEL_ID
        mediation["_model_revision"] = JUDGE_MODEL_REVISION
        mediation["_schema_version"] = JUDGE_SCHEMA_VERSION
        return mediation
