"""Replay up to four saved atomic visual questions, without running the pipeline."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question", action="append", required=True, help="sample_id:question_id")
    args = parser.parse_args()
    if not 1 <= len(args.question) <= 4 or len(set(args.question)) != len(args.question):
        raise ValueError("Use one to four distinct atomic questions")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.visual_grounding import VisualGroundingAgent
    from agents.visual_adapter import VisualQuestion
    from models.vision_model import Qwen3VLVisionModel
    from dataset.loaders import load_split, decode_image
    from engine.reproducibility import seed_stage
    from engine.judge_telemetry import monitor_run, mark_phase
    from run_figdebate import pipeline_source_checksum
    source, target = Path(args.source_run), Path(args.output)
    requests = [x.split(":", 1) for x in args.question]
    saved = {r["id"]:r for r in map(json.loads, (source/"records.jsonl").read_text(encoding="utf-8").splitlines())}
    config = json.loads((source/"run_config.json").read_text(encoding="utf-8"))
    rows = {r["id"]:r for r in load_split(config["dataset"]) if r["id"] in {x[0] for x in requests}}
    tasks = []
    for sample_id, question_id in requests:
        row, record = rows[sample_id], saved[sample_id]
        if (row["image_sha256"] != record["image_sha256"] or
                hashlib.sha256(row["caption"].encode()).hexdigest() != record["caption_sha256"]):
            raise ValueError("Saved question identity mismatch")
        old = next(a for a in record["trace"]["visual_output"]["_internal"]["atomic_answers"] if a["question_id"] == question_id)
        image = decode_image(row["image_bytes"])
        if question_id.startswith("initial_region_ocr_"):
            regions = {"initial_region_ocr_" + re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_"): crop
                       for name, crop in VisualGroundingAgent._text_relation_crops(image)}
            image = regions[question_id]
        diagnostics = old["generation_diagnostics"]
        tokens = diagnostics.get("primary", diagnostics).get("max_new_tokens", 96)
        tasks.append((sample_id, old, image, tokens))
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"source_run":str(source.resolve()), "questions":args.question,
                "pipeline_source_sha256":pipeline_source_checksum(), "scope":"atomic_visual_only"}
    (target/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with monitor_run(target):
        runtime = Qwen3VLVisionModel(hardware_profile="paper-8gb")
        agent = VisualGroundingAgent(runtime)
        for sample_id, old, image, tokens in tasks:
            seed_stage(config.get("seed", 42), sample_id, "visual_grounding")
            mark_phase("atomic_visual:" + old["question_id"])
            question = VisualQuestion(old["question_id"], old["question_type"], old["question"], tokens)
            result = asdict(agent._run_atomic_question(image, question))
            entry = {"id":sample_id, "old_status":old["status"], "old_error":old["error"],
                     "old_seconds":old["elapsed_seconds"], "result":result}
            with (target/"records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps({"id":sample_id,"status":result["status"],"error":result["error"],
                              "seconds":result["elapsed_seconds"]}), flush=True)
    (target/"complete.json").write_text(json.dumps({"completed":len(tasks)}), encoding="utf-8")


if __name__ == "__main__": main()
