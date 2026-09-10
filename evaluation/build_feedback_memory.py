"""Build frozen, evidence-backed feedback memory from development records."""

import argparse
import json
import os
from pathlib import Path
import pandas as pd

from engine.evidence_ledger import (
    RELATION_FOR_LABEL,
    attach_evidence_audit,
    build_evidence_ledger,
)
from engine.feedback_loop import FeedbackLoop


def build_memory(records_path, output_path):
    run_dir = os.path.dirname(os.path.abspath(records_path))
    output_dir = os.path.dirname(os.path.abspath(output_path))
    config_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(config_path):
        raise ValueError("Feedback requires verified development run configuration")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset_name = config.get("dataset")
    if config.get("selection_only"):
        raise ValueError("Selection-only plans are not completed model runs")
    if dataset_name not in {"vflute_train", "vflute_train_dev50", "vflute_val"} or config.get("data_usage", {}).get("purpose") != "diagnostic":
        raise ValueError("Feedback requires a declared diagnostic train/validation run; never test")
    records = [json.loads(line) for line in Path(records_path).read_text(encoding="utf8").splitlines() if line.strip()]
    from evaluation.input_coverage import reconcile
    _, audit = reconcile(pd.DataFrame(records), records_path)
    if audit["status"] != "RECONCILED" or audit["missing_ids"] or not audit.get("source_lineage_verified"):
        raise ValueError("Feedback requires complete outputs with verified source identities")
    if not audit.get("references_bound_to_manifest"):
        raise ValueError("Feedback requires reference explanations bound to the source manifest")
    # Check against the actual dataset, not only mutually consistent run files.
    from dataset.loaders import load_split
    from engine.sampling import select_from_manifest
    manifest = json.loads((Path(run_dir) / "sample_manifest.json").read_text(encoding="utf8"))
    select_from_manifest(load_split(dataset_name), manifest, config["requested_samples"])
    if os.path.exists(output_path):
        raise FileExistsError("Choose a new feedback output; existing memory is immutable")
    os.makedirs(output_dir, exist_ok=True)

    loop = FeedbackLoop(
        max_examples=100,
        log_file=os.path.join(output_dir, "feedback_memory_build_log.json"),
    )
    wrong = 0
    created = 0
    skipped_contract_or_duplicate = 0
    with open(records_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("prediction") == record.get("ground_truth"):
                continue
            wrong += 1
            trace = record.get("trace", {}) or {}
            visual = trace.get("visual_output", {}) or {}
            language = trace.get("language_output", {}) or {}
            comparison = trace.get("comparison", {}) or {}
            ledger = trace.get("evidence_ledger") or build_evidence_ledger(
                visual, language, comparison
            )
            relation = RELATION_FOR_LABEL.get(record.get("ground_truth"))
            failure_type = (
                "missed_grounded_support"
                if relation == "SUPPORT"
                else "missed_grounded_conflict"
            )
            decision = attach_evidence_audit(
                trace.get("final_decision", {}), ledger
            )
            context = {
                "visual_output": visual,
                "language_output": language,
                "comparison": comparison,
                "decision": decision,
                "evidence_ledger": ledger,
            }
            if loop.add_verified_case(
                context,
                record.get("ground_truth"),
                failure_type,
                loop.CALIBRATED_RULES[failure_type],
                {
                    "sample_id": record.get("id"),
                    "source_run": run_dir,
                    "source": "development_error_with_verified_evidence",
                },
            ):
                created += 1
            else:
                skipped_contract_or_duplicate += 1

    memories = loop.export_examples()
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(memories, handle, indent=2)
    summary = {
        "records_path": os.path.abspath(records_path),
        "incorrect_samples": wrong,
        "procedural_case_memories": created,
        "skipped_contract_or_duplicate": skipped_contract_or_duplicate,
        "output_path": os.path.abspath(output_path),
        "evidence_policy": (
            "gold_identifies_development_failure_but_memory_stores_no_label_direction"
        ),
    }
    with open(
        os.path.join(output_dir, "feedback_memory_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build evidence-backed feedback memory from a development run."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_memory(args.records, args.output), indent=2))


if __name__ == "__main__":
    main()
