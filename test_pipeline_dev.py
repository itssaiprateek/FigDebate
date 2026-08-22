"""
step3_run_dev_split_evaluation.py

Runs the full 3-agent pipeline (Agent 1 -> Agent 2 -> Comparator -> Arbiter)
on all 50 dev split samples, with checkpoint/resume logic so a crash never
loses prior work (per the project's own code cleanup checklist).

Produces:
    step3_agent1_checkpoint.jsonl        -- Agent 1 outputs, resumable
    step3_full_pipeline_results.jsonl    -- final per-sample results, resumable
    step3_dev_split_results.csv          -- clean final CSV for review
    printed macro F1 / CONTRADICTS recall vs. the baseline numbers

Safe to re-run: it will skip any sample ID already present in the
checkpoints and only process what's missing.
"""

import os
import csv
import json
import gc
import torch

from dataset.loaders import load_dev_split, decode_image
from models.model_loader import LlavaModel
from models.mistral_loader import MistralModel
from agents.agent1 import VisualGroundingAgent
from agents.agent2 import ClaimExtractionAgent
from comparators.comparator_v2 import compare
from arbiter.arbiter import Arbiter


AGENT1_CHECKPOINT = "step3_agent1_checkpoint.jsonl"
FINAL_CHECKPOINT = "step3_full_pipeline_results.jsonl"
FINAL_CSV = "step3_dev_split_results.csv"


# ======================================================
# Checkpoint helpers
# ======================================================

def load_checkpoint_ids(path):
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add(row["id"])
            except json.JSONDecodeError:
                continue
    return completed


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ======================================================
# Metrics
# ======================================================

def compute_metrics(results):
    """Binary macro F1 + per-class precision/recall for ENTAILS/CONTRADICTS."""

    labels = ["ENTAILS", "CONTRADICTS"]
    metrics = {}

    for label in labels:
        tp = sum(1 for r in results if r["true_label"] == label and r["predicted_label"] == label)
        fp = sum(1 for r in results if r["true_label"] != label and r["predicted_label"] == label)
        fn = sum(1 for r in results if r["true_label"] == label and r["predicted_label"] != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = (metrics["ENTAILS"]["f1"] + metrics["CONTRADICTS"]["f1"]) / 2
    accuracy = sum(1 for r in results if r["predicted_label"] == r["true_label"]) / len(results)

    metrics["macro_f1"] = macro_f1
    metrics["accuracy"] = accuracy

    return metrics


# ======================================================
# Main
# ======================================================

def main():

    dev = load_dev_split()
    print(f"Dev split loaded: {len(dev)} samples")

    # ------------------------------------------------------
    # STAGE 1 -- Agent 1 (LLaVA) on every sample, resumable
    # ------------------------------------------------------

    completed_agent1_ids = load_checkpoint_ids(AGENT1_CHECKPOINT)
    print(f"Agent 1 checkpoint: {len(completed_agent1_ids)}/{len(dev)} samples already done")

    remaining_for_agent1 = [s for s in dev if s["id"] not in completed_agent1_ids]

    if remaining_for_agent1:

        print(f"Running Agent 1 on {len(remaining_for_agent1)} remaining samples...")

        llava = LlavaModel()
        agent1 = VisualGroundingAgent(llava)

        for i, sample in enumerate(remaining_for_agent1):

            print(f"\n[{i + 1}/{len(remaining_for_agent1)}] Agent 1 on {sample['id']}")

            image = decode_image(sample["image_bytes"])
            output = agent1.analyze(image)

            append_jsonl(AGENT1_CHECKPOINT, {
                "id": sample["id"],
                "caption": sample["caption"],
                "true_label": sample["label"],
                "phenomenon": sample.get("phenomenon"),
                "agent1_output": output,
            })

            gc.collect()

        del agent1
        del llava
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            print(f"GPU memory after Agent 1 cleanup: {torch.cuda.memory_allocated()/1024**3:.3f} GB")

    else:
        print("All samples already have Agent 1 output -- skipping Stage 1 entirely.")

    # ------------------------------------------------------
    # STAGE 2 -- Agent 2 + Comparator + Arbiter, resumable
    # ------------------------------------------------------

    agent1_rows = load_jsonl(AGENT1_CHECKPOINT)
    agent1_by_id = {row["id"]: row for row in agent1_rows}

    completed_final_ids = load_checkpoint_ids(FINAL_CHECKPOINT)
    print(f"Final pipeline checkpoint: {len(completed_final_ids)}/{len(dev)} samples already done")

    remaining_for_final = [s for s in dev if s["id"] not in completed_final_ids]

    if remaining_for_final:

        print(f"Running Agent 2 + Arbiter on {len(remaining_for_final)} remaining samples...")

        mistral = MistralModel()
        agent2 = ClaimExtractionAgent(mistral.model, mistral.tokenizer)
        arbiter = Arbiter(mistral.model, mistral.tokenizer)

        for i, sample in enumerate(remaining_for_final):

            sample_id = sample["id"]

            if sample_id not in agent1_by_id:
                print(f"[SKIP] {sample_id} has no Agent 1 output yet -- re-run Stage 1 first.")
                continue

            print(f"\n[{i + 1}/{len(remaining_for_final)}] Agent 2 + Arbiter on {sample_id}")

            agent1_output = agent1_by_id[sample_id]["agent1_output"]

            agent2_output = agent2.analyze(sample["caption"])

            comparison = compare(
                agent1_output,
                agent2_output,
                caption=sample["caption"],
            )

            visual_grounding_text = (
                f"Visual Description: {agent1_output['visual_description']}\n"
                f"Objects: {agent1_output['objects']}\n"
                f"Scene Type: {agent1_output['scene_type']}\n"
                f"Symbolic Tone: {agent1_output['symbolic_tone']}"
            )

            language_understanding_text = (
                f"Surface Meaning: {agent2_output['surface_meaning']}\n"
                f"Figurative Type: {agent2_output['figurative_type']}\n"
                f"Intended Meaning: {agent2_output['intended_meaning']}\n"
                f"Background Knowledge: {agent2_output['background_knowledge']}"
            )

            arbiter_output = arbiter.analyze(
                caption=sample["caption"],
                visual_grounding=visual_grounding_text,
                language_understanding=language_understanding_text,
                comparison=comparison,
            )

            raw_text = arbiter_output["_internal"].get("raw_output", "") or ""
            hedge_detected = (
                "final decision" in raw_text.lower()
                and "none" in raw_text.lower().split("final decision")[-1][:50]
            )

            row = {
                "id": sample_id,
                "phenomenon": sample.get("phenomenon"),
                "true_label": sample["label"],
                "predicted_label": arbiter_output.get("label"),
                "confidence": arbiter_output.get("confidence"),
                "debate_needed": arbiter_output.get("debate_needed"),
                "label_was_forced": arbiter_output.get("_label_was_forced"),
                "figurative_type_predicted": agent2_output.get("figurative_type"),
                "hedge_detected": hedge_detected,
            }

            append_jsonl(FINAL_CHECKPOINT, row)

            gc.collect()

        del agent2
        del arbiter
        del mistral
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print("All samples already have final pipeline output -- skipping Stage 2 entirely.")

    # ------------------------------------------------------
    # STAGE 3 -- Metrics + CSV export
    # ------------------------------------------------------

    results = load_jsonl(FINAL_CHECKPOINT)

    if len(results) < len(dev):
        print(
            f"\n[WARNING] Only {len(results)}/{len(dev)} samples have final results. "
            "Re-run this script to resume -- it picks up exactly where it left off."
        )

    if not results:
        print("No results yet -- nothing to report.")
        return

    metrics = compute_metrics(results)

    forced_count = sum(1 for r in results if r["label_was_forced"])
    hedge_count = sum(1 for r in results if r["hedge_detected"])
    unknown_fig_type_count = sum(1 for r in results if r["figurative_type_predicted"] == "unknown")

    print("\n" + "=" * 60)
    print(f"FINAL METRICS -- {len(results)}/{len(dev)} dev split samples")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}   (baseline V-FLUTE macro F1 = 0.465 -- this is the number to beat)")
    print(
        f"ENTAILS      -- precision={metrics['ENTAILS']['precision']:.3f} "
        f"recall={metrics['ENTAILS']['recall']:.3f} f1={metrics['ENTAILS']['f1']:.3f}"
    )
    print(
        f"CONTRADICTS  -- precision={metrics['CONTRADICTS']['precision']:.3f} "
        f"recall={metrics['CONTRADICTS']['recall']:.3f} f1={metrics['CONTRADICTS']['f1']:.3f}   "
        f"(baseline recall = 0.05-0.12, project target >= 0.40 -- this is your headline number)"
    )
    print(f"\nLabels forced from raw text: {forced_count}/{len(results)}")
    print(f"Hedge ('Final Decision: None') detected: {hedge_count}/{len(results)}")
    print(f"figurative_type returned as 'unknown': {unknown_fig_type_count}/{len(results)}")

    fieldnames = [
        "id", "phenomenon", "true_label", "predicted_label", "confidence",
        "debate_needed", "label_was_forced", "figurative_type_predicted",
        "hedge_detected",
    ]

    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nSaved: {os.path.abspath(FINAL_CSV)}")
    print(
        f"Checkpoints (keep these until you trust the CSV, then safe to delete): "
        f"{AGENT1_CHECKPOINT}, {FINAL_CHECKPOINT}"
    )


if __name__ == "__main__":
    main()
