"""Paired comparison of two FigDebate prediction files."""

import argparse
import csv
import json
import math
import os


LABELS = ("ENTAILS", "CONTRADICTS")


def _truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _mean(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None


def _truth_rate(rows, key):
    if not any(str(row.get(key, "")).strip() for row in rows):
        return None
    return sum(_truth(row.get(key)) for row in rows) / len(rows)


def _conditional_truth_rate(rows, key, condition_key):
    selected = [row for row in rows if _truth(row.get(condition_key))]
    if not selected or not any(
        str(row.get(key, "")).strip() for row in selected
    ):
        return None
    return sum(_truth(row.get(key)) for row in selected) / len(selected)


def _directional_packet_rate(rows):
    values = [str(row.get("decision_packet_profile", "")).strip() for row in rows]
    if not any(values):
        return None
    directional = {
        "SUPPORT_ONLY", "CONFLICT_ONLY", "MIXED_DIRECTIONAL_EVIDENCE"
    }
    return sum(value in directional for value in values) / len(values)


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def _macro_f1(rows):
    scores = []
    for label in LABELS:
        tp = sum(row["ground_truth"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["ground_truth"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["ground_truth"] == label and row["prediction"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def _exact_mcnemar_p(control_only, treatment_only):
    discordant = control_only + treatment_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(control_only, treatment_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2 * tail)


def compare(control_path, treatment_path, output_dir):
    control = _read(control_path)
    treatment = _read(treatment_path)
    if set(control) != set(treatment):
        missing_control = sorted(set(treatment) - set(control))
        missing_treatment = sorted(set(control) - set(treatment))
        raise ValueError(
            "Runs do not contain identical sample IDs. "
            f"Missing from control={missing_control[:5]}, "
            f"missing from treatment={missing_treatment[:5]}"
        )

    paired = []
    for sample_id in sorted(control):
        left = control[sample_id]
        right = treatment[sample_id]
        if left["ground_truth"] != right["ground_truth"]:
            raise ValueError(f"Ground truth differs for {sample_id}")
        gold = left["ground_truth"]
        left_correct = left["prediction"] == gold
        right_correct = right["prediction"] == gold
        paired.append({
            "id": sample_id,
            "phenomenon": left.get("phenomenon", ""),
            "ground_truth": gold,
            "control_prediction": left["prediction"],
            "treatment_prediction": right["prediction"],
            "control_correct": left_correct,
            "treatment_correct": right_correct,
            "changed": left["prediction"] != right["prediction"],
            "outcome": (
                "corrected" if not left_correct and right_correct
                else "harmed" if left_correct and not right_correct
                else "unchanged"
            ),
            "control_feedback_rules": left.get("feedback_matched_rule_ids", ""),
            "treatment_feedback_rules": right.get("feedback_matched_rule_ids", ""),
            "control_feedback_agents": left.get(
                "feedback_matched_target_agents", ""
            ),
            "treatment_feedback_agents": right.get(
                "feedback_matched_target_agents", ""
            ),
            "control_feedback_mechanisms": left.get(
                "feedback_matched_failure_mechanisms", ""
            ),
            "treatment_feedback_mechanisms": right.get(
                "feedback_matched_failure_mechanisms", ""
            ),
        })

    total = len(paired)
    corrections = sum(row["outcome"] == "corrected" for row in paired)
    harms = sum(row["outcome"] == "harmed" for row in paired)
    control_accuracy = sum(row["control_correct"] for row in paired) / total
    treatment_accuracy = sum(row["treatment_correct"] for row in paired) / total
    control_rows = list(control.values())
    treatment_rows = list(treatment.values())
    metrics = {
        "samples": total,
        "control_accuracy": control_accuracy,
        "treatment_accuracy": treatment_accuracy,
        "accuracy_delta": treatment_accuracy - control_accuracy,
        "control_macro_f1": _macro_f1(control_rows),
        "treatment_macro_f1": _macro_f1(treatment_rows),
        "macro_f1_delta": _macro_f1(treatment_rows) - _macro_f1(control_rows),
        "prediction_changes": sum(row["changed"] for row in paired),
        "corrections": corrections,
        "harms": harms,
        "net_correct_decisions": corrections - harms,
        "control_debate_trigger_rate": sum(
            _truth(row.get("debate_triggered")) for row in control_rows
        ) / total,
        "treatment_debate_trigger_rate": sum(
            _truth(row.get("debate_triggered")) for row in treatment_rows
        ) / total,
        "control_feedback_acceptances": sum(
            _truth(row.get("feedback_revision_accepted")) for row in control_rows
        ),
        "treatment_feedback_acceptances": sum(
            _truth(row.get("feedback_revision_accepted")) for row in treatment_rows
        ),
        "control_feedback_match_rate": _truth_rate(
            control_rows, "feedback_memory_active"
        ),
        "treatment_feedback_match_rate": _truth_rate(
            treatment_rows, "feedback_memory_active"
        ),
        "control_directional_evidence_rate": sum(
            _truth(row.get("final_evidence_valid")) for row in control_rows
        ) / total,
        "treatment_directional_evidence_rate": sum(
            _truth(row.get("final_evidence_valid")) for row in treatment_rows
        ) / total,
        "control_claim_contract_valid_rate": _truth_rate(
            control_rows, "claim_contract_valid"
        ),
        "treatment_claim_contract_valid_rate": _truth_rate(
            treatment_rows, "claim_contract_valid"
        ),
        "control_literal_contract_valid_rate": _truth_rate(
            control_rows, "claim_literal_contract_valid"
        ),
        "treatment_literal_contract_valid_rate": _truth_rate(
            treatment_rows, "claim_literal_contract_valid"
        ),
        "control_pragmatic_activation_rate": _truth_rate(
            control_rows, "claim_pragmatic_activated"
        ),
        "treatment_pragmatic_activation_rate": _truth_rate(
            treatment_rows, "claim_pragmatic_activated"
        ),
        "control_required_binding_resolution_rate": _conditional_truth_rate(
            control_rows,
            "comparator_relation_binding_observed",
            "comparator_relation_binding_required",
        ),
        "treatment_required_binding_resolution_rate": _conditional_truth_rate(
            treatment_rows,
            "comparator_relation_binding_observed",
            "comparator_relation_binding_required",
        ),
        "control_mean_entity_state_bindings": _mean(
            control_rows, "agent1_entity_state_binding_count"
        ),
        "treatment_mean_entity_state_bindings": _mean(
            treatment_rows, "agent1_entity_state_binding_count"
        ),
        "control_directional_packet_rate": _directional_packet_rate(control_rows),
        "treatment_directional_packet_rate": _directional_packet_rate(treatment_rows),
        "control_level2_debate_rate": sum(
            _truth(row.get("debate_triggered"))
            and float(row.get("debate_level") or 0) >= 2
            for row in control_rows
        ) / total,
        "treatment_level2_debate_rate": sum(
            _truth(row.get("debate_triggered"))
            and float(row.get("debate_level") or 0) >= 2
            for row in treatment_rows
        ) / total,
        "control_recovery_evidence_yield": _conditional_truth_rate(
            control_rows,
            "debate_grounding_recovery_evidence_success",
            "debate_grounding_recovery_attempted",
        ),
        "treatment_recovery_evidence_yield": _conditional_truth_rate(
            treatment_rows,
            "debate_grounding_recovery_evidence_success",
            "debate_grounding_recovery_attempted",
        ),
        "control_review_board_grounded_rate": _truth_rate(
            control_rows, "review_board_directionally_grounded"
        ),
        "treatment_review_board_grounded_rate": _truth_rate(
            treatment_rows, "review_board_directionally_grounded"
        ),
        "control_mean_runtime_seconds": _mean(control_rows, "runtime_seconds"),
        "treatment_mean_runtime_seconds": _mean(treatment_rows, "runtime_seconds"),
        "mcnemar_exact_p_value": _exact_mcnemar_p(harms, corrections),
        "control_path": os.path.abspath(control_path),
        "treatment_path": os.path.abspath(treatment_path),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(
        os.path.join(output_dir, "changed_cases.csv"), "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(row for row in paired if row["changed"])
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("FIGDEBATE PAIRED RUN COMPARISON\n" + "=" * 40 + "\n\n")
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare matched FigDebate runs.")
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = compare(args.control, args.treatment, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
