"""Build frozen, evidence-backed feedback memory from development records."""

import argparse
import json
import os

from engine.evidence_ledger import attach_evidence_audit, build_evidence_ledger
from engine.feedback_loop import FeedbackLoop


def build_memory(records_path, output_path):
    run_dir = os.path.dirname(os.path.abspath(records_path))
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            dataset_name = json.load(handle).get("dataset")
        if dataset_name == "vflute_test":
            raise ValueError("Feedback memory must never be built from vflute_test.")

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
            calibration = loop.calibration_rule(
                language,
                comparison,
                decision,
                record.get("ground_truth"),
                record.get("phenomenon"),
            )
            if calibration is None:
                skipped_contract_or_duplicate += 1
                continue
            target_agent, failure_type, advice = calibration
            if loop.add_verified_case(
                context,
                record.get("ground_truth"),
                failure_type,
                advice,
                {
                    "sample_id": record.get("id"),
                    "source_run": run_dir,
                    "source": "development_error_with_verified_evidence",
                },
                target_agent=target_agent,
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
