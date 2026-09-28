"""Bounded paired replay of initial reasoning on frozen upstream outputs.

Development diagnostics only. Does not rerun upstream witnesses or expose gold
labels/references to inference. Four variants differ only in the named mechanisms.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    from project_environment import configure
    configure()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from models.language_model import MistralModel
    from arbiter.arbiter import Arbiter
    from engine.reproducibility import seed_stage
    from engine.runtime_accounting import begin_accounting, set_scope, sample_accounting
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs", nargs="+", required=True, type=Path)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=["baseline","completion","grounding","combined"], default=["baseline","completion","grounding","combined"])
    args = parser.parse_args()
    if len(args.sample_ids) > 20 or len(set(args.sample_ids)) != len(args.sample_ids):
        raise ValueError("Choose at most 20 unique development cases")
    rows = {}
    sources = []
    for directory in args.source_runs:
        data = (directory / "records.jsonl").read_bytes()
        sources.append({"run": str(directory), "records_sha256": hashlib.sha256(data).hexdigest()})
        for line in data.decode("utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in args.sample_ids:
                if row["id"] in rows:
                    raise ValueError("Duplicate source case")
                rows[row["id"]] = row
    if set(rows) != set(args.sample_ids):
        raise ValueError("Missing selected cases")
    args.output.mkdir(parents=True, exist_ok=False)
    from engine.run_provenance import snapshot_source
    snapshot_source(str(Path(__file__).resolve().parents[1]), str(args.output))
    manifest = {"purpose": "paired_initial_stage_replay_not_full_pipeline_or_heldout", "seed": 42,
                "sources": sources, "ids": args.sample_ids, "feedback": "disabled",
                "gold_and_references": "post_prediction_evaluation_only", "variants": args.variants, "order": "case_major_rotating_arms"}
    from run_figdebate import pipeline_source_checksum
    manifest["pipeline_source_sha256"] = pipeline_source_checksum()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runtime = MistralModel()
    begin_accounting()
    variants = [(n, c, g) for n, c, g in [("baseline",False,False),("completion",True,False),
                 ("grounding",False,True),("combined",True,True)] if n in args.variants]
    for index, identifier in enumerate(args.sample_ids):
        offset = index % len(variants)
        for name, completion, grounding in variants[offset:] + variants[:offset]:
            arbiter = Arbiter(runtime.model, runtime.tokenizer, completion_checks=completion, grounded_interpretation=grounding)
            row = rows[identifier]
            trace = row["trace"]
            seed_stage(42, identifier, "initial_reasoning")
            set_scope(identifier + ":" + name, "initial_reasoning")
            started = time.perf_counter()
            # Deliberate allowlist: neither references, labels nor prior decisions enter.
            decision = arbiter.analyze(trace["comparison"]["caption"], deepcopy(trace["visual_output"]),
                                      deepcopy(trace["language_output"]), deepcopy(trace["comparison"]))
            result = {"id": identifier, "variant": name, "decision": decision,
                      "wall_seconds": time.perf_counter()-started,
                      "runtime": sample_accounting(identifier + ":" + name),
                      "gold": row["ground_truth"], "correct": decision.get("label") == row["ground_truth"],
                      "reference_explanation": row.get("reference_explanation"),
                      "saved_initial_label": row["initial_prediction"]}
            with (args.output / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False)+"\n")
            print(json.dumps({k:result[k] for k in ("id","variant","correct","wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
