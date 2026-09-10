"""Finite, real-tokenizer decoder qualification. Never load model weights."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from engine.output_contracts import CORE, INTERPRETATION, object_schema, schema_for_contract
from engine.structured_decoder import DECODER_ID, prefix_constraint


def example(schema):
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema["type"]
    if isinstance(kind, list):
        return None if "null" in kind else example(dict(schema, type=kind[0]))
    if kind == "object":
        return {key: example(value) for key, value in schema["properties"].items()}
    if kind == "array":
        return [example(schema["items"])]
    return {"string": "A complete sentence.", "number": 0.5, "boolean": True}[kind]


def accepts(tokenizer, schema, text):
    allowed = prefix_constraint(tokenizer, schema)
    prefix = tokenizer.encode("Qualification prompt", add_special_tokens=False)
    for index, token in enumerate(tokenizer.encode(text, add_special_tokens=False)):
        if token not in allowed(0, torch.tensor(prefix)):
            return False, {"index": index, "token": token, "text": tokenizer.decode([token])}
        prefix.append(token)
    return tokenizer.eos_token_id in allowed(0, torch.tensor(prefix)), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    from models.hub_source import cached_snapshot_or_hub
    path, kwargs = cached_snapshot_or_hub("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a")
    schemas = {"core": CORE, "interpretation": INTERPRETATION}
    schemas["core_extraction"] = object_schema({key: value for key, value in CORE["properties"].items()
        if key not in {"expected_visual_state", "opposite_visual_state"}})
    from engine.claim_witness import SCHEMA as witness_schema
    schemas["witness"] = witness_schema
    from engine.claim_semantics import ANSWER_SCHEMA, COMPARISON_SCHEMA, RELATION_SCHEMA
    schemas.update(source_answer=ANSWER_SCHEMA, source_comparison=COMPARISON_SCHEMA,
                   condition_relation=RELATION_SCHEMA)
    from engine.claim_graph import SCHEMA as graph_schema
    schemas["claim_graph"] = graph_schema
    for name in ("independent_candidate", "independent_verification", "tribunal_review"):
        schemas[name] = schema_for_contract(name)
    cases = [(name, schema, json.dumps(example(schema)), True) for name, schema in schemas.items()]
    short = object_schema({"text": {"type": "string", "maxLength": 80},
                           "items": {"type": "array", "items": {"type": "boolean"}}})
    for index, value in enumerate(['Sentence.', 'Quote: "yes".', 'Line\nnext\tvalue', 'caf\u00e9 \u732b', '', '\\path\\']):
        cases.append((f"boundary_{index}", short, json.dumps(dict(text=value, items=[True, False]), ensure_ascii=False), True))
    for index, value in enumerate(['{}', '{"text":null,"items":[]}', '{"text":"ok","items":[3]}',
                                   json.dumps(dict(text='x' * 81, items=[])), '{"text":"ok","items":[],"extra":1}']):
        cases.append((f"illegal_{index}", short, value, False))
    results = []
    from models.vision_model import local_vision_model_path
    for name, location, options in [("mistral", path, kwargs), ("qwen_judge", "models/judge/Qwen3.5-4B", {}),
                                    ("qwen_vision", str(local_vision_model_path()), {})]:
        tokenizer = AutoTokenizer.from_pretrained(location, **dict(options, local_files_only=True))
        for case, schema, value, expected in cases:
            started = time.perf_counter()
            actual, failure = accepts(tokenizer, schema, value)
            results.append(dict(tokenizer=name, case=case, expected=expected, accepted=actual,
                                passed=actual == expected, failure=failure, seconds=time.perf_counter()-started))
        print(json.dumps({"tokenizer": name, "passed": all(row["passed"] for row in results if row["tokenizer"] == name)}), flush=True)
    report = dict(decoder=DECODER_ID, cases=results, passed=all(row["passed"] for row in results),
                  source_sha256=hashlib.sha256(Path("engine/structured_decoder.py").read_bytes()).hexdigest(),
                  scope="finite_tokenizer_conformance_not_live_model_or_semantic_qualification")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
