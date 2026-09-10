"""Non-mutating pipeline diagnosis. Synthetic probes are NOT accuracy evaluations.

Writes a new diagnostic artifact only; does not alter production parameters/data.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from agents.claim_extraction import ClaimExtractionAgent
from engine.claim_semantics import CHECKS, audit_core
from engine.claim_witness import audit_claim_witness
from engine.debate import DebateEngine
from engine.output_contracts import CORE, object_schema, saturated_text_fields, validate_shape


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def cpu_probes(rows):
    from lmformatenforcer import JsonSchemaParser
    results = {}
    agent = object.__new__(ClaimExtractionAgent)
    approval = dict(**{key: True for key in CHECKS}, _format_valid=True)
    # Fault injection isolates the deterministic gate from the model's fallibility.
    witness = dict(stance="ENDORSE", support_requirement="None", conflict_requirement="None",
                   figurative_mechanism="unknown", ambiguity="none", reason="test")
    with patch.object(agent, "_generate_section", return_value=(witness, json.dumps(witness), 0, {"schema_valid": True})), \
         patch("engine.claim_witness.audit_core", return_value=approval):
        result = audit_claim_witness(agent, "The line rises.", "Claim subject: line")
    results["sentinel_requirements_with_false_positive_auditor"] = {
        "accepted": result["requirements_valid"], "errors": result["requirement_errors"]}
    schema = object_schema({"reason": {"type": "string", "maxLength": 20}})
    parser = JsonSchemaParser(schema)
    for char in '{"reason":"' + 'x' * 20:
        assert char in parser.get_allowed_characters(), repr(char)
        parser = parser.add_character(char)
    results["character_limit"] = {"allowed_at_limit": parser.get_allowed_characters(),
                                  "can_finish_word": "a" in parser.get_allowed_characters()}
    results["clause_detector"] = {
        "short_dangling_clause": saturated_text_fields('{"reason":"because the"}', schema),
        "complete_unpunctuated_20_chars": saturated_text_fields('{"reason":"The sky appears blue"}', schema)}
    prompts = []
    for row in rows:
        lang = row["trace"]["language_output"]
        prior = DebateEngine.build_agent2_challenge_prompt(None, lang, {})
        recovered = {key: agent._audit_field(prior, key.replace("_", " ")) for key in CORE["properties"]}
        prompts.append(dict(id=row["id"], source_lists={key: lang.get(key) for key in ("negation", "quantities", "claim_modifiers")},
                            recovered_lists={key: recovered[key] for key in ("negation", "quantities", "claim_modifiers")}))
    results["prompt_round_trip"] = prompts
    reviews = []
    for row in rows:
        for value in walk(row["trace"]["judge"]):
            raw = value.get("_raw_output", "")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if "best_semantic_judgment" not in parsed:
                continue
            reviews.append(dict(id=row["id"], error=value.get("_format_error"),
                fields={key: {"length": len(parsed[key]), "tail": parsed[key][-45:]} for key in
                        ("visual_premise", "caption_premise", "semantic_bridge", "counter_interpretation", "reason") if key in parsed}))
    results["saved_review_boundaries"] = reviews
    return results


def live_probes(rows, auditor="mistral"):
    from models.language_model import MistralModel
    from engine.reproducibility import seed_stage
    if auditor == "qwen":
        from models.judge_model import QwenJudgeModel
        runtime = QwenJudgeModel(hardware_profile="paper-8gb")
        tokenizer = runtime.processor.tokenizer
    else:
        runtime = MistralModel()
        tokenizer = runtime.tokenizer
    agent = ClaimExtractionAgent(runtime.model, tokenizer)
    row = next(row for row in rows if row["id"] == "vflute_train_4315")
    lang = row["trace"]["language_output"]
    caption = lang["original_caption"]
    core = {key: copy.deepcopy(lang.get(key)) for key in CORE["properties"]}
    core["expected_visual_state"] = "The faculty meeting is calm and relaxing."
    core["opposite_visual_state"] = "The faculty meeting is chaotic and tense."
    variants = [("complete_opposition", core)]
    missing = copy.deepcopy(core)
    missing["expected_visual_state"] = "None"
    variants.append(("missing_support", missing))
    swapped = copy.deepcopy(core)
    swapped["expected_visual_state"], swapped["opposite_visual_state"] = core["opposite_visual_state"], core["expected_visual_state"]
    variants.append(("swapped_direction", swapped))
    identical = copy.deepcopy(core)
    identical["opposite_visual_state"] = identical["expected_visual_state"]
    variants.append(("identical_states", identical))
    results = []
    for name, fields in variants:
        seed_stage(42, "critical_diagnostic", "controlled_audit")
        audit = audit_core(agent, caption, fields)
        result = dict(probe=name, caption=caption, fields=fields, audit=audit)
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--auditor", choices=("mistral", "qwen"), default="mistral")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Diagnostic artifacts are immutable")
    source = Path(args.source) / "records.jsonl"
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    result = dict(purpose="root_cause_diagnosis_not_accuracy_evaluation", source=str(source.resolve()),
                  source_sha256=hashlib.sha256(raw).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpu=cpu_probes(rows))
    target.parent.mkdir(parents=True, exist_ok=True)
    # Preserve CPU results even if a live probe fails.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["cpu"]), flush=True)
    if args.live:
        result["auditor"] = args.auditor
        result["live"] = live_probes(rows, args.auditor)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
