"""Live judge-only diagnostic replay of frozen saved upstream states.

This is not a full pipeline evaluation. Gold is read only after review/resolution.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence-view", choices=("first", "current"), default="current")
    args = parser.parse_args()
    from dataset.loaders import load_split, decode_image
    from models.judge_model import QwenJudgeModel
    from agents.multimodal_judge import TribunalMediatorAgent
    from engine.tribunal import apply_tribunal_resolution
    from engine.reproducibility import seed_stage
    from engine.review_outcome import review_timing
    from evaluation.tribunal_quality import summarize_tribunal
    from engine.runtime_accounting import begin_accounting, sample_accounting, set_scope
    baseline = Path(args.baseline_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "records.jsonl").exists():
        raise FileExistsError("Use a new output directory for each frozen candidate")
    wanted = list(dict.fromkeys(args.sample_ids))
    rows = {r["id"]: r for r in (json.loads(line) for line in (baseline / "records.jsonl").read_text(encoding="utf-8").splitlines()) if r["id"] in wanted}
    config = json.loads((baseline / "run_config.json").read_text())
    raw = {r["id"]: r for r in load_split(config["dataset"]) if r["id"] in wanted}
    if set(raw) != set(wanted) or set(rows) != set(wanted):
        raise ValueError("Requested cases missing from dataset or saved run")
    root = Path(__file__).resolve().parents[1]
    source_paths = [p for directory in ("agents", "arbiter", "comparators", "dataset", "engine", "models", "evaluation", "utils", "tests")
                    for p in sorted((root / directory).rglob("*.py"))
                    if not {"data", ".venv", "__pycache__", ".git"}.intersection(p.relative_to(root).parts)]
    source_paths += list(root.glob("*.py")) + [root / "requirements.txt"]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for rel, expected in source_hashes.items():
        data = (root / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Source changed while preparing qualification")
        target = out / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / "manifest.json").write_text(json.dumps({"sample_ids": wanted, "source_hashes": source_hashes,
        "baseline_run": str(baseline), "mode": "judge_only_frozen_latest_upstream_state",
        "evidence_view": args.evidence_view, "not_full_pipeline": True}, indent=2))
    started = time.perf_counter()
    runtime = QwenJudgeModel(hardware_profile="paper-8gb")
    runtime.hardware_profile = replace(runtime.hardware_profile, judge_evidence_first=args.evidence_view == "first")
    reviewer = TribunalMediatorAgent(runtime)
    results = []
    begin_accounting()
    for identifier in wanted:
        row, item = rows[identifier], raw[identifier]
        trace = deepcopy(row["trace"])
        # Remove old tribunal-derived evidence and its descendants; no prior
        # judge certification is permitted to support the new judge.
        ledger = trace["evidence_ledger"]
        excluded = {x["id"] for x in ledger if x.get("source") in {"semantic_bridge_verifier", "tribunal_relation_verifier"}}
        while True:
            more = {x["id"] for x in ledger if set(x.get("derived_from_ids", [])) & excluded}
            if more <= excluded:
                break
            excluded |= more
        ledger = [x for x in ledger if x["id"] not in excluded]
        seed_stage(42, identifier, "tribunal_review", 1)
        set_scope(identifier, "tribunal_review", 1)
        beginning = time.perf_counter()
        print(f"START {identifier}", flush=True)
        language, debate = trace["language_output"], trace["debate_details"]
        review = reviewer.review(decode_image(item["image_bytes"]), item["caption"], trace["visual_output"],
            language, trace["comparison"], ledger, debate, current_decision=trace["initial_decision"],
            pre_hearing=trace.get("pre_hearing", {}))
        decision, final_ledger, resolution = apply_tribunal_resolution(trace["initial_decision"], review, ledger,
            language.get("claim_contract", {}), agent2_requirements_valid=debate.get("agent2_requirements_valid", True),
            agent1_critique=debate.get("agent1_critique", {}), agent2_critique=debate.get("agent2_critique", {}),
            semantic_bridge_mode="corroborated", language_output=language, source_caption=item["caption"])
        result = dict(id=identifier, initial_prediction=row["initial_prediction"], prediction=decision["label"],
            ground_truth=item["label"], judge_verdict=review.get("best_semantic_judgment", "ABSTAIN"),
            judge_format_valid=review.get("_format_valid", False), judge_format_error=review.get("_format_error", ""),
            tribunal_revision_reason=resolution["reason"], runtime_seconds=time.perf_counter()-beginning,
            tribunal_timing=[review_timing(review)],
            trace={"judge": {"tribunal_reviews": [review], "tribunal_resolution": resolution},
                   "runtime_accounting": sample_accounting(identifier)})
        results.append(result)
        with (out / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        (out / "quality.json").write_text(json.dumps(summarize_tribunal(results, time.perf_counter()-started), indent=2))
        print(f"DONE {identifier} initial={result['initial_prediction']} judge={result['judge_verdict']} final={result['prediction']} gold={result['ground_truth']} seconds={result['runtime_seconds']:.1f} reason={resolution['reason']}", flush=True)
    print(json.dumps(summarize_tribunal(results, time.perf_counter()-started), indent=2), flush=True)


if __name__ == "__main__":
    main()
