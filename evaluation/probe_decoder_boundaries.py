"""CPU token-level probes of the production grammar adapter; no model inference."""
import json
from pathlib import Path
import torch
from lmformatenforcer import JsonSchemaParser
from transformers import AutoTokenizer
from engine.output_contracts import prefix_constraint, schema_for_contract
from engine.claim_semantics import SCHEMA, CHECKS


def character_check(schema, text):
    parser = JsonSchemaParser(schema)
    for index, char in enumerate(text):
        if char not in parser.get_allowed_characters():
            return dict(accepted=False, offset=index, character=char)
        parser = parser.add_character(char)
    return dict(accepted=parser.can_end(), scope="complete_character_sequence_not_token_enumeration")


def check(tokenizer, schema, text, stop_after=None):
    allowed = prefix_constraint(tokenizer, schema)
    prefix = tokenizer.encode("Diagnostic prompt", add_special_tokens=True)
    for index, token in enumerate(tokenizer.encode(text, add_special_tokens=False)):
        options = allowed(0, torch.tensor(prefix))
        if token not in options:
            return dict(accepted=False, offset=index, token=token,
                        token_text=tokenizer.decode([token]), decoded_prefix=tokenizer.decode(prefix))
        prefix.append(token)
        if stop_after is not None and index >= stop_after:
            return dict(accepted=True, scope="prefix_through_originally_blocked_token", checked_tokens=index + 1)
    return dict(accepted=True, scope="complete_sequence")


def main():
    from models.hub_source import cached_snapshot_or_hub
    mistral, kwargs = cached_snapshot_or_hub("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a")
    paths = [("mistral", mistral, kwargs), ("qwen_judge", "models/judge/Qwen3.5-4B", {})]
    result = {}
    for name, path, kwargs in paths:
        tokenizer = AutoTokenizer.from_pretrained(path, **dict(kwargs, local_files_only=True))
        schema = schema_for_contract("tribunal_review")
        obj = dict(best_semantic_judgment="ENTAILS", relation="SUPPORT", admissibility="PLAUSIBLE",
                   semantic_bridge_type=schema["properties"]["semantic_bridge_type"]["enum"][0],
                   visual_premise="The line rises.", caption_premise="The line rises.", semantic_bridge="The directions match.",
                   counter_interpretation="None established.", reason="Matching directions.", evidence_ids=["VF001"],
                   confidence=0.5, counter_interpretation_strength=0.2, requested_follow_up="NONE")
        probes = {"short_complete_review": check(tokenizer, schema, json.dumps(obj))}
        # Exact grammar path without exhaustive vocabulary enumeration. The previous
        # exhaustive shortcut-disabled Mistral test succeeded, but Qwen was too slow.
        probes["same_review_character_grammar"] = character_check(schema, json.dumps(obj))
        for value in (True, False):
            obj = {key: value for key in CHECKS}
            obj.update(defective_fields=[], reason="Diagnostic fixture.")
            probes[f"all_booleans_{value}"] = check(tokenizer, SCHEMA, json.dumps(obj))
        result[name] = probes
        print(json.dumps({name: probes}), flush=True)
    target = Path("outputs/critical_decoder_probe_20260907_d.json")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
