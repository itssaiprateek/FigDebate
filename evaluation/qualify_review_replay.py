"""Live review-stage replay on fixed dev hearings, not a fresh full-pipeline run."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from agents.multimodal_judge import TribunalMediatorAgent
from dataset.loaders import load_split, decode_image
from engine.reproducibility import seed_stage
from engine.runtime_accounting import begin_accounting, sample_accounting
from models.judge_model import QwenJudgeModel
from run_figdebate import pipeline_source_checksum, append_record


def replay(source_run, output, sample_ids=None):
    source = Path(source_run)
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    if config["dataset"] != "vflute_train_dev50":
        raise ValueError("This diagnostic replay is restricted to existing dev50 hearings")
    target = Path(output)
    if target.exists():
        raise FileExistsError("Replay outputs are immutable")
    records_bytes = (source / "records.jsonl").read_bytes()
    rows = [json.loads(line) for line in records_bytes.decode().splitlines() if line.strip()]
    if sample_ids is not None:
        if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= {row['id'] for row in rows}:
            raise ValueError("Replay IDs must be unique and present in the source run")
        selected = set(sample_ids)
        rows = [row for row in rows if row['id'] in selected]
    data = {row["id"]: row for row in load_split(config["dataset"])}
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate replay cases")
    for row in rows:
        raw = data[row["id"]]
        if row["image_sha256"] != raw["image_sha256"] or row["caption_sha256"] != hashlib.sha256(raw["caption"].encode()).hexdigest():
            raise ValueError("Replay input identity mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = pipeline_source_checksum()
    begin_accounting()
    started = time.perf_counter()
    runtime = QwenJudgeModel(hardware_profile="paper-8gb")
    agent = TribunalMediatorAgent(runtime)
    for row in rows:
        raw, trace = data[row["id"]], row["trace"]
        seed_stage(42, row["id"], "review_contract_replay")
        # No prior judge text, baseline decision, gold or reference explanation
        # is passed. Only the recorded hearing/evidence and unchanged inputs.
        review = agent.review(decode_image(raw["image_bytes"]), raw["caption"],
            trace["visual_output"], trace["language_output"], {}, trace["evidence_ledger"],
            trace["debate_details"], current_decision=None)
        record = dict(id=row["id"], purpose="review_stage_replay_not_full_pipeline_or_accuracy_study",
                      source_records_sha256=hashlib.sha256(records_bytes).hexdigest(),
                      pipeline_source_sha256=source_hash, source_run=str(source.resolve()),
                      image_sha256=row["image_sha256"], caption_sha256=row["caption_sha256"],
                      hardware_profile=runtime.hardware_profile.as_dict(), review=review,
                      runtime_accounting=sample_accounting(row["id"]),
                      cumulative_wall_seconds=time.perf_counter() - started)
        append_record(str(target), record)
        print(json.dumps(dict(id=row["id"], valid=review.get("_format_valid"),
                             error=review.get("_format_error"), relation=review.get("relation"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ids", nargs='+', help="Exact source-run IDs for a bounded diagnostic replay")
    args = parser.parse_args()
    replay(args.source_run, args.output, args.sample_ids)
