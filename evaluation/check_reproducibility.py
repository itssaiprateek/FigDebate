"""Compare shared samples across two runs and locate first-stage divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import pandas as pd


STAGES = (
    ("visual_grounding", (
        "agent1_visible_text", "agent1_symbolic_tone", "agent1_schema_complete",
        "agent1_ocr_usable", "agent1_relation_binding_present",
    )),
    ("claim_contract", (
        "agent2_caption_proposition", "claim_relation_family",
        "claim_contract_valid", "claim_contract_warnings",
    )),
    ("initial_decision", (
        "initial_prediction", "semantic_entails_score",
        "semantic_contradicts_score", "comparator_evidence_status",
    )),
    ("debate", (
        "agent1_critique_response_status", "agent1_critique_claim_relation",
        "agent2_requirements_valid", "agent2_support_requirement",
        "agent2_conflict_requirement",
    )),
    ("tribunal", (
        "judge_verdict", "semantic_bridge_type",
        "semantic_bridge_verification_status", "tribunal_revision_reason",
        "tribunal_revision_accepted",
    )),
    ("final_decision", ("final_prediction", "decision_method")),
)


def _normal(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return " ".join(text.split())


def compare(left_path, right_path, output_dir):
    left = pd.read_csv(left_path).set_index("id", drop=False)
    right = pd.read_csv(right_path).set_index("id", drop=False)
    shared = sorted(set(left.index) & set(right.index))
    rows = []
    for sample_id in shared:
        first_stage = ""
        differences = []
        for stage, fields in STAGES:
            stage_differences = [
                field for field in fields
                if field in left and field in right
                and _normal(left.at[sample_id, field])
                != _normal(right.at[sample_id, field])
            ]
            if stage_differences and not first_stage:
                first_stage = stage
            differences.extend(stage_differences)
        rows.append({
            "id": sample_id,
            "identical": not differences,
            "first_divergent_stage": first_stage,
            "different_fields": " | ".join(differences),
        })
    report = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    report.to_csv(os.path.join(output_dir, "reproducibility_comparison.csv"), index=False)
    summary = {
        "schema_version": "1.0",
        "shared_samples": len(shared),
        "identical_samples": int(report["identical"].sum()) if not report.empty else 0,
        "mismatched_samples": int((~report["identical"]).sum()) if not report.empty else 0,
        "shared_sample_ids_sha256": hashlib.sha256(
            "\n".join(shared).encode("utf-8")
        ).hexdigest(),
        "first_divergent_stage_distribution": (
            report.loc[~report["identical"], "first_divergent_stage"]
            .value_counts().to_dict() if not report.empty else {}
        ),
    }
    with open(os.path.join(output_dir, "reproducibility_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    summary = compare(args.left, args.right, args.output_dir)
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["mismatched_samples"] else 0)


if __name__ == "__main__":
    main()
