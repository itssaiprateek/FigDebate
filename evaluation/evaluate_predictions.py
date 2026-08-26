"""Evaluation and paper-audit artifact generation for FigDebate runs."""

import argparse
import json
import os
import re

import pandas as pd
try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
except ImportError:
    from evaluation.metrics_core import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )


LABELS = ("ENTAILS", "CONTRADICTS")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FigDebate predictions.")
    parser.add_argument("--input", required=True, help="Path to predictions CSV.")
    parser.add_argument("--output-dir", help="Directory for metrics; defaults to the CSV directory.")
    return parser.parse_args()


def as_bool(series):
    return series.fillna(False).astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def distribution(series):
    return {str(key): int(value) for key, value in series.fillna("missing").value_counts().items()}


def metric_row(group, group_name, value):
    valid = group[group["_valid"]]
    row = {group_name: value, "samples": int(len(group)), "valid_predictions": int(len(valid))}
    if valid.empty:
        return {**row, "accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    balanced = None
    if valid["ground_truth"].nunique() == len(LABELS):
        balanced = float(
            balanced_accuracy_score(valid["ground_truth"], valid["prediction"])
        )
    return {
        **row,
        "accuracy": float(accuracy_score(valid["ground_truth"], valid["prediction"])),
        "balanced_accuracy": balanced,
        "macro_f1": float(f1_score(valid["ground_truth"], valid["prediction"], labels=LABELS, average="macro", zero_division=0)),
    }


def confidence_metrics(valid):
    if "final_confidence" not in valid.columns:
        return {"samples": 0, "mean": None, "median": None, "p95": None, "brier_score": None, "ece_10_bin": None}, pd.DataFrame()
    values = numeric(valid["final_confidence"])
    usable = valid.loc[values.notna()].copy()
    usable["confidence"] = values.loc[usable.index].clip(0, 1)
    if usable.empty:
        return {"samples": 0, "mean": None, "median": None, "p95": None, "brier_score": None, "ece_10_bin": None}, pd.DataFrame()
    usable["p_entails"] = usable["confidence"].where(usable["prediction"] == "ENTAILS", 1 - usable["confidence"])
    usable["entails_target"] = (usable["ground_truth"] == "ENTAILS").astype(float)
    usable["correct"] = usable["prediction"] == usable["ground_truth"]
    usable["confidence_bin"] = pd.cut(usable["confidence"], bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    rows, ece = [], 0.0
    for band, group in usable.groupby("confidence_bin", observed=False):
        if group.empty:
            continue
        mean_confidence, empirical_accuracy = float(group["confidence"].mean()), float(group["correct"].mean())
        ece += (len(group) / len(usable)) * abs(mean_confidence - empirical_accuracy)
        rows.append({"confidence_band": str(band), "samples": int(len(group)), "mean_confidence": mean_confidence, "accuracy": empirical_accuracy, "calibration_gap": abs(mean_confidence - empirical_accuracy)})
    return {
        "samples": int(len(usable)), "mean": float(usable["confidence"].mean()),
        "median": float(usable["confidence"].median()), "p95": float(usable["confidence"].quantile(.95)),
        "brier_score": float(((usable["p_entails"] - usable["entails_target"]) ** 2).mean()), "ece_10_bin": float(ece),
    }, pd.DataFrame(rows)


def runtime_profile(df):
    columns = ("agent1_seconds", "agent2_seconds", "claim_retry_seconds", "comparator_seconds", "evidence_verifier_seconds", "arbiter_primary_seconds", "citation_retry_seconds", "feedback_review_seconds", "format_retry_seconds", "binary_resolution_seconds", "debate_seconds", "judge_seconds", "mediator_seconds", "runtime_seconds")
    rows = []
    for column in columns:
        if column not in df:
            continue
        values = numeric(df[column]).dropna()
        if not values.empty:
            rows.append({"stage": column, "samples": int(len(values)), "mean_seconds": float(values.mean()), "median_seconds": float(values.median()), "p95_seconds": float(values.quantile(.95)), "total_seconds": float(values.sum())})
    return pd.DataFrame(rows)


def _explanation_tokens(value):
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def _lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, start=1):
            current.append(
                previous[index - 1] + 1
                if left == right
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def explanation_metrics(df):
    if not {"reference_explanation", "final_reason"}.issubset(df.columns):
        return {
            "samples": 0, "mean_token_f1": None, "mean_rouge_l_f1": None,
            "note": "Reference explanations were unavailable.",
        }, pd.DataFrame()
    rows = []
    for index, row in df.iterrows():
        reference = _explanation_tokens(row.get("reference_explanation"))
        generated = _explanation_tokens(row.get("final_reason"))
        if not reference or not generated:
            token_f1 = 0.0
            rouge_l = 0.0
        else:
            reference_set, generated_set = set(reference), set(generated)
            overlap = len(reference_set & generated_set)
            precision = overlap / len(generated_set)
            recall = overlap / len(reference_set)
            token_f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            lcs = _lcs_length(reference, generated)
            rouge_precision = lcs / len(generated)
            rouge_recall = lcs / len(reference)
            rouge_l = (
                2 * rouge_precision * rouge_recall
                / (rouge_precision + rouge_recall)
                if rouge_precision + rouge_recall else 0.0
            )
        rows.append({
            "sample": row.get("sample", index),
            "id": row.get("id"),
            "ground_truth": row.get("ground_truth"),
            "prediction": row.get("prediction"),
            "token_f1": token_f1,
            "rouge_l_f1": rouge_l,
            "reference_explanation": row.get("reference_explanation"),
            "generated_explanation": row.get("final_reason"),
        })
    output = pd.DataFrame(rows)
    return {
        "samples": int(len(output)),
        "mean_token_f1": float(output["token_f1"].mean()),
        "mean_rouge_l_f1": float(output["rouge_l_f1"].mean()),
        "note": (
            "Lexical diagnostics against V-FLUTE reference explanations; "
            "they are not substitutes for human faithfulness review."
        ),
    }, output


def evaluate_predictions(input_path, output_dir=None):
    """Evaluate one predictions.csv and write all paper-ready artifacts."""
    # Consolidate the wide run table before adding audit columns. This avoids
    # pandas fragmentation warnings without changing any metric values.
    df = pd.read_csv(input_path).copy()
    missing = {"ground_truth", "prediction", "phenomenon"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    output_dir = os.path.abspath(output_dir or os.path.dirname(os.path.abspath(input_path)))
    os.makedirs(output_dir, exist_ok=True)
    df["_valid"] = df["prediction"].isin(LABELS)
    if "final_decision_valid" in df:
        df["_valid"] &= as_bool(df["final_decision_valid"])
    valid, invalid = df[df["_valid"]].copy(), df[~df["_valid"]].copy()
    if valid.empty:
        raise ValueError("No valid ENTAILS or CONTRADICTS predictions were produced.")

    y_true, y_pred = valid["ground_truth"], valid["prediction"]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    per_label = {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS)}
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

    debate_mask = as_bool(df["debate_triggered"]) if "debate_triggered" in df else pd.Series(False, index=df.index)
    if {"initial_prediction", "final_prediction"}.issubset(df.columns):
        debated = df[debate_mask & df["initial_prediction"].isin(LABELS) & df["final_prediction"].isin(LABELS)].copy()
    else:
        debated = pd.DataFrame()
    if not debated.empty:
        debated["initial_correct"] = debated["initial_prediction"] == debated["ground_truth"]
        debated["final_correct"] = debated["final_prediction"] == debated["ground_truth"]
        debated["outcome"] = "unchanged"
        debated.loc[~debated["initial_correct"] & debated["final_correct"], "outcome"] = "corrected"
        debated.loc[debated["initial_correct"] & ~debated["final_correct"], "outcome"] = "harmed"
    correction_rate = float((debated.get("outcome", pd.Series(dtype=str)) == "corrected").mean()) if not debated.empty else 0.0
    harm_rate = float((debated.get("outcome", pd.Series(dtype=str)) == "harmed").mean()) if not debated.empty else 0.0
    debate_corrections = int((debated.get("outcome", pd.Series(dtype=str)) == "corrected").sum()) if not debated.empty else 0
    debate_harms = int((debated.get("outcome", pd.Series(dtype=str)) == "harmed").sum()) if not debated.empty else 0
    debate_net_gain = debate_corrections - debate_harms
    proposal_column = (
        "debate_unconstrained_proposed_label"
        if "debate_unconstrained_proposed_label" in df.columns
        else "debate_proposed_label"
    )
    if proposal_column in df.columns:
        proposals = df[
            debate_mask & df[proposal_column].isin(LABELS)
        ].copy()
        proposals["proposal_correct"] = (
            proposals[proposal_column] == proposals["ground_truth"]
        )
        if "initial_prediction" in proposals:
            proposals["proposal_outcome"] = "unchanged"
            initially_correct = (
                proposals["initial_prediction"] == proposals["ground_truth"]
            )
            proposals.loc[
                ~initially_correct & proposals["proposal_correct"],
                "proposal_outcome",
            ] = "corrected"
            proposals.loc[
                initially_correct & ~proposals["proposal_correct"],
                "proposal_outcome",
            ] = "harmed"
    else:
        proposals = pd.DataFrame()

    judge_requested_mask = (
        as_bool(df["judge_requested"])
        if "judge_requested" in df else pd.Series(False, index=df.index)
    )
    mediated_requested_mask = (
        judge_requested_mask
        & df.get("judge_mode", pd.Series("", index=df.index)).eq("mediated")
    )
    effective_contract_valid = (
        as_bool(df.get("judge_format_valid", pd.Series(False, index=df.index)))
    )
    if "mediator_format_valid" in df:
        effective_contract_valid.loc[mediated_requested_mask] = as_bool(
            df.loc[mediated_requested_mask, "mediator_format_valid"]
        )
    effective_verdict = df.get(
        "judge_verdict", pd.Series("", index=df.index, dtype=object)
    ).astype(object).fillna("")
    if "mediator_provisional_verdict" in df:
        effective_verdict.loc[mediated_requested_mask] = df.loc[
            mediated_requested_mask, "mediator_provisional_verdict"
        ]
    judge_columns = (
        "sample", "id", "phenomenon", "ground_truth", "pre_judge_prediction",
        "prediction", "judge_mode", "judge_scope", "judge_requested",
        "judge_status", "judge_trigger_reasons", "judge_verdict",
        "judge_confidence", "judge_format_valid", "judge_evidence_ids",
        "judge_invalid_evidence_ids", "judge_reason",
        "judge_revision_accepted", "judge_revision_reason", "judge_seconds",
        "judge_feedback_candidate_recorded", "judge_feedback_role",
        "judge_feedback_memory_update_applied",
        "mediator_status", "mediator_provisional_verdict",
        "mediator_confidence", "mediator_format_valid",
        "mediator_evidence_ids", "mediator_invalid_evidence_ids",
        "mediator_disputed_issues", "mediator_agent1_questions",
        "mediator_agent2_questions", "mediator_verification_requests",
        "mediator_reason", "mediator_usable", "mediator_tiebreak_used",
        "mediator_seconds",
    )
    judge_df = df[[name for name in judge_columns if name in df]].copy()
    if {"pre_judge_prediction", "judge_verdict"}.issubset(df.columns):
        judge_comparable = df[
            judge_requested_mask
            & df["pre_judge_prediction"].isin(LABELS)
            & df["judge_verdict"].isin(LABELS)
        ].copy()
        judge_comparable["pre_judge_correct"] = (
            judge_comparable["pre_judge_prediction"]
            == judge_comparable["ground_truth"]
        )
        judge_comparable["judge_counterfactual_correct"] = (
            judge_comparable["judge_verdict"] == judge_comparable["ground_truth"]
        )
        judge_comparable["judge_counterfactual_outcome"] = "unchanged"
        judge_comparable.loc[
            ~judge_comparable["pre_judge_correct"]
            & judge_comparable["judge_counterfactual_correct"],
            "judge_counterfactual_outcome",
        ] = "corrected"
        judge_comparable.loc[
            judge_comparable["pre_judge_correct"]
            & ~judge_comparable["judge_counterfactual_correct"],
            "judge_counterfactual_outcome",
        ] = "harmed"
    else:
        judge_comparable = pd.DataFrame()
    judge_counterfactual_corrections = int(
        judge_comparable.get(
            "judge_counterfactual_outcome", pd.Series(dtype=str)
        ).eq("corrected").sum()
    ) if not judge_comparable.empty else 0
    judge_counterfactual_harms = int(
        judge_comparable.get(
            "judge_counterfactual_outcome", pd.Series(dtype=str)
        ).eq("harmed").sum()
    ) if not judge_comparable.empty else 0
    if not judge_comparable.empty:
        judge_df["judge_counterfactual_outcome"] = judge_comparable[
            "judge_counterfactual_outcome"
        ].reindex(judge_df.index)

    judge_accepted_mask = (
        as_bool(df["judge_revision_accepted"])
        if "judge_revision_accepted" in df else pd.Series(False, index=df.index)
    )
    judge_accepted = df[
        judge_accepted_mask
        & df.get("pre_judge_prediction", pd.Series(index=df.index)).isin(LABELS)
        & df["prediction"].isin(LABELS)
    ].copy()
    if not judge_accepted.empty:
        accepted_before_correct = (
            judge_accepted["pre_judge_prediction"] == judge_accepted["ground_truth"]
        )
        accepted_after_correct = (
            judge_accepted["prediction"] == judge_accepted["ground_truth"]
        )
        judge_accepted_corrections = int(
            (~accepted_before_correct & accepted_after_correct).sum()
        )
        judge_accepted_harms = int(
            (accepted_before_correct & ~accepted_after_correct).sum()
        )
    else:
        judge_accepted_corrections = 0
        judge_accepted_harms = 0

    confidence, confidence_df = confidence_metrics(valid)
    explanation, explanation_df = explanation_metrics(valid)
    phenomenon_df = pd.DataFrame([metric_row(group, "phenomenon", name) for name, group in df.groupby("phenomenon", dropna=False)])
    if "figurative_type_predicted" in df:
        type_accuracy = df.assign(_correct_type=df["figurative_type_predicted"].astype(str).str.lower() == df["phenomenon"].astype(str).str.lower()).groupby("phenomenon")["_correct_type"].mean()
        phenomenon_df["caption_type_phenomenon_agreement"] = phenomenon_df["phenomenon"].map(type_accuracy)
    decision_method_df = pd.DataFrame([metric_row(group, "decision_method", name) for name, group in df.groupby("decision_method", dropna=False)]) if "decision_method" in df else pd.DataFrame()
    comparator_df = pd.DataFrame([metric_row(group, "comparator_evidence_status", name) for name, group in df.groupby("comparator_evidence_status", dropna=False)]) if "comparator_evidence_status" in df else pd.DataFrame()
    runtime_df = runtime_profile(df)
    errors = df[df["ground_truth"] != df["prediction"]].copy()
    error_columns = ("sample", "id", "phenomenon", "ground_truth", "prediction", "final_confidence", "semantic_entails_score", "semantic_contradicts_score", "semantic_neutral_score", "semantic_verifier_model", "decision_method", "comparator_evidence_status", "comparator_recommendation", "debate_triggered", "debate_trigger_reason", "debate_revision_accepted", "debate_unconstrained_proposed_label", "debate_relation_status", "debate_deficiencies", "agent1_critique_method", "agent1_critique_response_status", "agent1_critique_parser_errors", "agent1_critique_observed_entity", "agent1_critique_observed_state", "agent1_critique_claim_relation", "agent1_region_pairs", "agent2_critique_format_valid", "agent2_requirements_valid", "agent2_requirement_errors", "arbiter_relation_status", "arbiter_deficiencies", "claim_retry_attempted", "claim_retry_success", "feedback_mode", "feedback_memory_active", "figurative_type_predicted", "agent1_visible_text", "agent1_symbolic_tone", "agent1_schema_complete", "agent1_schema_issues", "agent1_targeted_recovery_attempted", "agent1_targeted_recovery_success", "arbiter_evidence_assessment", "final_reason")
    errors = errors[[name for name in error_columns if name in errors]].copy()
    grounding_columns = ("sample", "id", "phenomenon", "agent1_visible_text", "agent1_visible_text_count", "agent1_visual_fact_count", "agent1_visual_relation_count", "agent1_symbolic_tone", "agent1_visual_metaphor_count", "agent1_schema_complete", "agent1_schema_format_valid", "agent1_factual_grounding_present", "agent1_ocr_usable", "agent1_relation_binding_present", "agent1_schema_issues", "agent1_schema_retry_attempted", "agent1_schema_retry_success", "agent1_targeted_recovery_attempted", "agent1_targeted_recovery_success", "agent1_targeted_recovery_reason", "agent1_targeted_recovery_seconds", "agent1_visual_confidence", "agent1_seconds", "agent2_linguistic_cue", "agent2_polarity_reversal", "agent2_caption_proposition", "claim_retry_attempted", "claim_retry_success", "claim_retry_seconds", "agent2_language_confidence", "agent2_seconds", "arbiter_evidence_assessment")
    grounding_df = df[[name for name in grounding_columns if name in df]].copy()
    evidence_columns = (
        "sample", "id", "phenomenon", "ground_truth", "prediction",
        "evidence_ledger_count", "evidence_support_count", "evidence_conflict_count",
        "evidence_anchor_count", "debate_visual_evidence_count",
        "debate_visual_evidence_ids", "initial_evidence_status",
        "initial_evidence_valid",
        "initial_cited_evidence_ids", "final_evidence_status", "final_evidence_valid",
        "final_cited_evidence_ids", "debate_triggered", "debate_revision_accepted",
        "initial_source_evidence_valid", "initial_source_cited_evidence_ids",
        "final_source_evidence_valid", "final_source_cited_evidence_ids",
        "debate_proposed_evidence_status", "debate_proposed_evidence_valid",
        "debate_proposed_cited_evidence_ids", "feedback_matched_rule_ids",
        "debate_proposed_source_evidence_valid",
        "debate_proposed_source_cited_evidence_ids",
        "evidence_verifier_candidate_count", "evidence_verifier_verified_count",
        "evidence_verifier_support_count", "evidence_verifier_conflict_count",
        "evidence_verifier_neutral_count", "evidence_verifier_model",
        "evidence_verifier_revision", "evidence_verifier_seconds",
        "targeted_region_verification_method",
        "targeted_region_verification_decision_grade",
        "targeted_region_verification_reason",
    )
    evidence_df = df[[name for name in evidence_columns if name in df]].copy()
    feedback_columns = (
        "sample", "id", "phenomenon", "ground_truth", "prediction", "correct",
        "feedback_mode", "feedback_enabled", "feedback_available_rule_count",
        "feedback_memory_active", "feedback_matched_rule_count",
        "feedback_matched_rule_ids", "feedback_candidate_recorded",
        "feedback_update_applied", "feedback_failure_type", "feedback_target_agent",
        "feedback_matched_rule_scores", "feedback_baseline_prediction",
        "feedback_candidate_prediction", "feedback_post_review_prediction",
        "feedback_revision_accepted", "feedback_revision_reason",
        "feedback_decision_changed", "feedback_review_seconds",
    )
    feedback_df = df[[name for name in feedback_columns if name in df]].copy()
    feedback_active = as_bool(df["feedback_memory_active"]) if "feedback_memory_active" in df else pd.Series(False, index=df.index)
    feedback_performance = {
        "memory_inactive": metric_row(df[~feedback_active], "group", "memory_inactive"),
        "memory_active": metric_row(df[feedback_active], "group", "memory_active"),
    }
    if {
        "feedback_baseline_prediction", "feedback_post_review_prediction"
    }.issubset(df.columns):
        feedback_comparable = df[
            df["feedback_baseline_prediction"].isin(LABELS)
            & df["feedback_post_review_prediction"].isin(LABELS)
        ].copy()
        feedback_comparable["baseline_correct"] = (
            feedback_comparable["feedback_baseline_prediction"]
            == feedback_comparable["ground_truth"]
        )
        feedback_comparable["post_review_correct"] = (
            feedback_comparable["feedback_post_review_prediction"]
            == feedback_comparable["ground_truth"]
        )
        feedback_comparable["feedback_outcome"] = "unchanged"
        feedback_comparable.loc[
            ~feedback_comparable["baseline_correct"]
            & feedback_comparable["post_review_correct"],
            "feedback_outcome",
        ] = "corrected"
        feedback_comparable.loc[
            feedback_comparable["baseline_correct"]
            & ~feedback_comparable["post_review_correct"],
            "feedback_outcome",
        ] = "harmed"
    else:
        feedback_comparable = pd.DataFrame()
    if not feedback_comparable.empty:
        feedback_df["feedback_outcome"] = feedback_comparable[
            "feedback_outcome"
        ].reindex(feedback_df.index)
    feedback_active = (
        as_bool(feedback_comparable["feedback_memory_active"])
        | as_bool(feedback_comparable["feedback_revision_accepted"])
    ) if (
        not feedback_comparable.empty
        and "feedback_memory_active" in feedback_comparable
        and "feedback_revision_accepted" in feedback_comparable
    ) else pd.Series(False, index=feedback_comparable.index)
    feedback_corrections = int(
        (
            feedback_active
            & feedback_comparable.get(
                "feedback_outcome", pd.Series(dtype=str)
            ).eq("corrected")
        ).sum()
    ) if not feedback_comparable.empty else 0
    feedback_harms = int(
        (
            feedback_active
            & feedback_comparable.get(
                "feedback_outcome", pd.Series(dtype=str)
            ).eq("harmed")
        ).sum()
    ) if not feedback_comparable.empty else 0
    run_timing = {}
    run_timing_path = os.path.join(output_dir, "run_timing.json")
    if os.path.exists(run_timing_path):
        with open(run_timing_path, "r", encoding="utf-8") as handle:
            run_timing = json.load(handle)

    total = int(len(df))
    type_agreement = float((df["figurative_type_predicted"].astype(str).str.lower() == df["phenomenon"].astype(str).str.lower()).mean()) if "figurative_type_predicted" in df else None
    prediction_distribution = distribution(df["prediction"])
    entails_prediction_rate = float((df["prediction"] == "ENTAILS").mean())
    semantic_entails = numeric(df["semantic_entails_score"]) if "semantic_entails_score" in df else pd.Series(dtype=float)
    semantic_conflicts = numeric(df["semantic_contradicts_score"]) if "semantic_contradicts_score" in df else pd.Series(dtype=float)
    semantic_neutral = numeric(df["semantic_neutral_score"]) if "semantic_neutral_score" in df else pd.Series(dtype=float)
    region_review_mask = df["agent1_critique_method"].fillna("").eq("region_ocr") if "agent1_critique_method" in df else pd.Series(False, index=df.index)
    visual_retry_mask = as_bool(df["agent1_critique_format_retry_used"]) if "agent1_critique_format_retry_used" in df else pd.Series(False, index=df.index)
    visual_retry_success_mask = as_bool(df["agent1_critique_format_retry_success"]) if "agent1_critique_format_retry_success" in df else pd.Series(False, index=df.index)
    visual_consensus_mask = as_bool(df["visual_evidence_consensus_applied"]) if "visual_evidence_consensus_applied" in df else pd.Series(False, index=df.index)
    targeted_nli_accepted_mask = df["decision_method"].fillna("").eq("targeted_region_verifier") if "decision_method" in df else pd.Series(False, index=df.index)
    targeted_nli_attempt_mask = df["debate_proposed_decision_method"].fillna("").eq("targeted_region_verifier") if "debate_proposed_decision_method" in df else targeted_nli_accepted_mask
    grounding_schema = as_bool(df["agent1_schema_complete"]) if "agent1_schema_complete" in df else pd.Series(False, index=df.index)
    metrics = {
        "samples": total, "valid_prediction_count": int(len(valid)), "invalid_prediction_count": int(len(invalid)), "invalid_prediction_rate": float(len(invalid) / total),
        "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1, "per_label": per_label,
        "confusion_matrix": {"labels": list(LABELS), "matrix": cm.tolist()},
        "ground_truth_distribution": distribution(df["ground_truth"]), "prediction_distribution": prediction_distribution,
        "label_balance": {
            "entails_prediction_rate": entails_prediction_rate,
            "contradicts_prediction_rate": 1.0 - entails_prediction_rate,
            "absolute_prediction_imbalance": abs(entails_prediction_rate - 0.5),
        },
        "caption_type_distribution": distribution(df["figurative_type_predicted"]) if "figurative_type_predicted" in df else {},
        "caption_type_phenomenon_agreement": type_agreement,
        "caption_type_note": "Diagnostic agreement only: the dataset phenomenon may occur in the image, caption, or both.",
        "claim_relations": {
            "resolved_rate": float(as_bool(df["claim_relation_resolved"]).mean()) if "claim_relation_resolved" in df else None,
            "contract_valid_rate": float(as_bool(df["claim_contract_valid"]).mean()) if "claim_contract_valid" in df else None,
            "proposition_preserved_rate": float(as_bool(df["claim_contract_proposition_preserved"]).mean()) if "claim_contract_proposition_preserved" in df else None,
            "entity_frame_preserved_rate": float(as_bool(df["claim_contract_entity_frame_preserved"]).mean()) if "claim_contract_entity_frame_preserved" in df else None,
            "contract_warning_distribution": distribution(df.loc[df["claim_contract_warnings"].fillna("").ne(""), "claim_contract_warnings"]) if "claim_contract_warnings" in df else {},
            "family_distribution": distribution(df["claim_relation_family"]) if "claim_relation_family" in df else {},
            "polarity_distribution": distribution(df["claim_relation_polarity"]) if "claim_relation_polarity" in df else {},
            "structured_candidate_count": int(numeric(df["structured_relation_candidate_count"]).sum()) if "structured_relation_candidate_count" in df else 0,
            "retry_rate": float(as_bool(df["claim_retry_attempted"]).mean()) if "claim_retry_attempted" in df else None,
            "retry_success_rate": float(as_bool(df.loc[as_bool(df["claim_retry_attempted"]), "claim_retry_success"]).mean()) if "claim_retry_attempted" in df and as_bool(df["claim_retry_attempted"]).any() else None,
        },
        "confidence": confidence,
        "explanations": explanation,
        "decision_reliability": {
            "primary_arbiter_valid_rate": float(as_bool(df["primary_decision_valid"]).mean()) if "primary_decision_valid" in df else None,
            "format_retry_rate": float(as_bool(df["format_retry_used"]).mean()) if "format_retry_used" in df else None,
            "citation_retry_rate": float(as_bool(df["citation_retry_used"]).mean()) if "citation_retry_used" in df else None,
            "binary_resolution_rate": float(as_bool(df["binary_resolution_used"]).mean()) if "binary_resolution_used" in df else None,
            "forced_label_rate": float(as_bool(df["forced_label"]).mean()) if "forced_label" in df else None,
            "decision_method_distribution": distribution(df["decision_method"]) if "decision_method" in df else {},
            "review_board_status_distribution": distribution(df["review_board_status"]) if "review_board_status" in df else {},
            "review_board_binary_valid_rate": float(as_bool(df["review_board_binary_valid"]).mean()) if "review_board_binary_valid" in df else None,
            "review_board_directionally_grounded_rate": float(as_bool(df["review_board_directionally_grounded"]).mean()) if "review_board_directionally_grounded" in df else None,
            "review_board_confidence_cap_rate": float(as_bool(df["review_board_confidence_cap_applied"]).mean()) if "review_board_confidence_cap_applied" in df else None,
        },
        "comparator": {"evidence_status_distribution": distribution(df["comparator_evidence_status"]) if "comparator_evidence_status" in df else {}, "recommendation_distribution": distribution(df["comparator_recommendation"]) if "comparator_recommendation" in df else {}, "relation_binding_required_count": int(as_bool(df["comparator_relation_binding_required"]).sum()) if "comparator_relation_binding_required" in df else 0, "relation_binding_observed_count": int(as_bool(df["comparator_relation_binding_observed"]).sum()) if "comparator_relation_binding_observed" in df else 0, "text_surface_without_ocr_count": int(as_bool(df["comparator_text_surface_without_ocr"]).sum()) if "comparator_text_surface_without_ocr" in df else 0, "symbolic_object_candidate_count": int(as_bool(df["comparator_has_symbolic_object_candidate"]).sum()) if "comparator_has_symbolic_object_candidate" in df else 0},
        "semantic_scoring": {
            "mean_entails_probability": float(semantic_entails.mean()) if semantic_entails.notna().any() else None,
            "mean_contradicts_probability": float(semantic_conflicts.mean()) if semantic_conflicts.notna().any() else None,
            "mean_neutral_probability": float(semantic_neutral.mean()) if semantic_neutral.notna().any() else None,
            "verifier_model_distribution": distribution(df["semantic_verifier_model"]) if "semantic_verifier_model" in df else {},
        },
        "grounding": {
            "schema_complete_rate": float(grounding_schema.mean()),
            "format_valid_rate": float(as_bool(df["agent1_schema_format_valid"]).mean()) if "agent1_schema_format_valid" in df else None,
            "factual_grounding_rate": float(as_bool(df["agent1_factual_grounding_present"]).mean()) if "agent1_factual_grounding_present" in df else None,
            "ocr_usable_rate": float(as_bool(df["agent1_ocr_usable"]).mean()) if "agent1_ocr_usable" in df else None,
            "relation_binding_rate": float(as_bool(df["agent1_relation_binding_present"]).mean()) if "agent1_relation_binding_present" in df else None,
            "schema_retry_rate": float(as_bool(df["agent1_schema_retry_attempted"]).mean()) if "agent1_schema_retry_attempted" in df else None,
            "schema_retry_success_rate": float(as_bool(df.loc[as_bool(df["agent1_schema_retry_attempted"]), "agent1_schema_retry_success"]).mean()) if "agent1_schema_retry_attempted" in df and as_bool(df["agent1_schema_retry_attempted"]).any() else None,
            "targeted_recovery_attempt_count": int(as_bool(df["agent1_targeted_recovery_attempted"]).sum()) if "agent1_targeted_recovery_attempted" in df else 0,
            "targeted_recovery_success_rate": float(as_bool(df.loc[as_bool(df["agent1_targeted_recovery_attempted"]), "agent1_targeted_recovery_success"]).mean()) if "agent1_targeted_recovery_attempted" in df and as_bool(df["agent1_targeted_recovery_attempted"]).any() else None,
            "mean_visible_text_items": float(numeric(df["agent1_visible_text_count"]).mean()) if "agent1_visible_text_count" in df else None,
            "mean_visual_fact_items": float(numeric(df["agent1_visual_fact_count"]).mean()) if "agent1_visual_fact_count" in df else None,
            "mean_visual_relation_items": float(numeric(df["agent1_visual_relation_count"]).mean()) if "agent1_visual_relation_count" in df else None,
            "symbolic_evidence_rate": float(as_bool(df["comparator_has_symbolic_evidence"]).mean()) if "comparator_has_symbolic_evidence" in df else None,
            "mean_visual_metaphor_items": float(numeric(df["agent1_visual_metaphor_count"]).mean()) if "agent1_visual_metaphor_count" in df else None,
        },
        "evidence_provenance": {
            "final_status_distribution": distribution(df["final_evidence_status"]) if "final_evidence_status" in df else {},
            "direct_evidence_valid_rate": float(as_bool(df["final_evidence_valid"]).mean()) if "final_evidence_valid" in df else None,
            "grounded_source_citation_rate": float(as_bool(df["final_source_evidence_valid"]).mean()) if "final_source_evidence_valid" in df else None,
            "mean_ledger_entries": float(numeric(df["evidence_ledger_count"]).mean()) if "evidence_ledger_count" in df else None,
            "atomic_candidate_count": int(numeric(df["evidence_verifier_candidate_count"]).sum()) if "evidence_verifier_candidate_count" in df else 0,
            "atomic_verified_count": int(numeric(df["evidence_verifier_verified_count"]).sum()) if "evidence_verifier_verified_count" in df else 0,
            "atomic_verification_rate": float(numeric(df["evidence_verifier_verified_count"]).sum() / numeric(df["evidence_verifier_candidate_count"]).sum()) if "evidence_verifier_candidate_count" in df and numeric(df["evidence_verifier_candidate_count"]).sum() > 0 else None,
        },
        "debate": {"trigger_rate": float(debate_mask.mean()), "trigger_count": int(debate_mask.sum()), "trigger_reason_distribution": distribution(df.loc[debate_mask, "debate_trigger_reason"]) if "debate_trigger_reason" in df else {}, "level_distribution": distribution(df.loc[debate_mask, "debate_level"]) if "debate_level" in df else {}, "mean_need_score": float(numeric(df.loc[debate_mask, "debate_need_score"]).mean()) if debate_mask.any() and "debate_need_score" in df else None, "comparable_cases": int(len(debated)), "correction_count": debate_corrections, "harm_count": debate_harms, "net_correct_decisions": debate_net_gain, "correction_rate": correction_rate, "harm_rate": harm_rate, "proposal_count": int(len(proposals)), "proposal_accuracy": float(proposals["proposal_correct"].mean()) if not proposals.empty else None, "proposal_correction_count": int(proposals.get("proposal_outcome", pd.Series(dtype=str)).eq("corrected").sum()) if not proposals.empty else 0, "proposal_harm_count": int(proposals.get("proposal_outcome", pd.Series(dtype=str)).eq("harmed").sum()) if not proposals.empty else 0, "agent1_response_status_distribution": distribution(df.loc[debate_mask, "agent1_critique_response_status"]) if "agent1_critique_response_status" in df else {}, "agent2_requirement_error_distribution": distribution(df.loc[debate_mask & ~as_bool(df["agent2_requirements_valid"]), "agent2_requirement_errors"]) if "agent2_requirements_valid" in df and "agent2_requirement_errors" in df else {}, "arbiter_relation_status_distribution": distribution(df.loc[debate_mask, "debate_relation_status"]) if "debate_relation_status" in df else {}, "deficiency_distribution": distribution(df.loc[debate_mask & df["debate_deficiencies"].fillna("").ne(""), "debate_deficiencies"]) if "debate_deficiencies" in df else {}, "revision_acceptance_rate": float(as_bool(df.loc[debate_mask, "debate_revision_accepted"]).mean()) if debate_mask.any() and "debate_revision_accepted" in df else None, "visual_critique_format_valid_rate": float(as_bool(df.loc[debate_mask, "agent1_critique_format_valid"]).mean()) if debate_mask.any() and "agent1_critique_format_valid" in df else None, "visual_critique_retry_count": int(visual_retry_mask.sum()), "visual_critique_retry_success_rate": float(visual_retry_success_mask[visual_retry_mask].mean()) if visual_retry_mask.any() else None, "visual_evidence_consensus_count": int(visual_consensus_mask.sum()), "linguistic_critique_format_valid_rate": float(as_bool(df.loc[debate_mask, "agent2_critique_format_valid"]).mean()) if debate_mask.any() and "agent2_critique_format_valid" in df else None, "review_status_distribution": distribution(df.loc[debate_mask, "debate_review_status"]) if "debate_review_status" in df else {}, "visual_reinspection_evidence_count": int(numeric(df["debate_visual_evidence_count"]).sum()) if "debate_visual_evidence_count" in df else 0, "visual_reinspection_evidence_case_count": int((numeric(df["debate_visual_evidence_count"]) > 0).sum()) if "debate_visual_evidence_count" in df else 0, "region_ocr_review_count": int(region_review_mask.sum()), "targeted_region_verifier_attempt_count": int(targeted_nli_attempt_mask.sum()), "targeted_region_verifier_accepted_count": int(targeted_nli_accepted_mask.sum()), "targeted_region_verifier_proposal_accuracy": float((df.loc[targeted_nli_attempt_mask, "debate_proposed_label"] == df.loc[targeted_nli_attempt_mask, "ground_truth"]).mean()) if targeted_nli_attempt_mask.any() and "debate_proposed_label" in df else None, "targeted_region_verifier_accuracy": float((df.loc[targeted_nli_accepted_mask, "prediction"] == df.loc[targeted_nli_accepted_mask, "ground_truth"]).mean()) if targeted_nli_accepted_mask.any() else None},
        "feedback": {"mode_distribution": distribution(df["feedback_mode"]) if "feedback_mode" in df else {}, "role_distribution": distribution(df["feedback_role"]) if "feedback_role" in df else {}, "memory_active_samples": int(as_bool(df["feedback_memory_active"]).sum()) if "feedback_memory_active" in df else 0, "memory_match_rate": float(as_bool(df["feedback_memory_active"]).mean()) if "feedback_memory_active" in df else 0.0, "matched_rule_distribution": distribution(df.loc[df["feedback_matched_rule_count"].fillna(0).astype(float) > 0, "feedback_matched_rule_ids"]) if "feedback_matched_rule_count" in df and "feedback_matched_rule_ids" in df else {}, "revision_acceptance_count": int(as_bool(df["feedback_revision_accepted"]).sum()) if "feedback_revision_accepted" in df else 0, "correction_count": feedback_corrections, "harm_count": feedback_harms, "net_correct_decisions": feedback_corrections - feedback_harms, "candidate_count": int(as_bool(df["feedback_candidate_recorded"]).sum()) if "feedback_candidate_recorded" in df else 0, "update_count": int(as_bool(df["feedback_update_applied"]).sum()) if "feedback_update_applied" in df else 0, "failure_distribution": distribution(df["feedback_failure_type"].dropna()) if "feedback_failure_type" in df else {}, "performance": feedback_performance},
        "judge": {
            "mode_distribution": distribution(df["judge_mode"]) if "judge_mode" in df else {},
            "requested_count": int(judge_requested_mask.sum()),
            "requested_rate": float(judge_requested_mask.mean()),
            "status_distribution": distribution(df.loc[judge_requested_mask, "judge_status"]) if "judge_status" in df else {},
            "contract_valid_rate": float(effective_contract_valid[judge_requested_mask].mean()) if judge_requested_mask.any() else None,
            "verdict_distribution": distribution(effective_verdict[judge_requested_mask]),
            "mediator_usable_rate": float(as_bool(df.loc[mediated_requested_mask, "mediator_usable"]).mean()) if mediated_requested_mask.any() and "mediator_usable" in df else None,
            "mediated_tiebreak_count": int(as_bool(df["mediator_tiebreak_used"]).sum()) if "mediator_tiebreak_used" in df else 0,
            "comparable_binary_judgments": int(len(judge_comparable)),
            "counterfactual_corrections": judge_counterfactual_corrections,
            "counterfactual_harms": judge_counterfactual_harms,
            "counterfactual_net_correct": judge_counterfactual_corrections - judge_counterfactual_harms,
            "accepted_revisions": int(judge_accepted_mask.sum()),
            "accepted_corrections": judge_accepted_corrections,
            "accepted_harms": judge_accepted_harms,
            "accepted_net_correct": judge_accepted_corrections - judge_accepted_harms,
            "feedback_review_candidates": int(as_bool(df["judge_feedback_candidate_recorded"]).sum()) if "judge_feedback_candidate_recorded" in df else 0,
            "feedback_memory_updates": int(as_bool(df["judge_feedback_memory_update_applied"]).sum()) if "judge_feedback_memory_update_applied" in df else 0,
        },
        "runtime": runtime_df.set_index("stage").to_dict(orient="index") if not runtime_df.empty else {},
        "run_timing": run_timing,
    }

    pd.DataFrame(cm, index=LABELS, columns=LABELS).rename_axis("ground_truth").to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    phenomenon_df.to_csv(os.path.join(output_dir, "phenomenon_breakdown.csv"), index=False)
    decision_method_df.to_csv(os.path.join(output_dir, "decision_method_breakdown.csv"), index=False)
    comparator_df.to_csv(os.path.join(output_dir, "comparator_analysis.csv"), index=False)
    debated.to_csv(os.path.join(output_dir, "debate_analysis.csv"), index=False)
    proposals.to_csv(
        os.path.join(output_dir, "debate_proposal_analysis.csv"), index=False
    )
    feedback_df.to_csv(os.path.join(output_dir, "feedback_analysis.csv"), index=False)
    judge_df.to_csv(os.path.join(output_dir, "judge_analysis.csv"), index=False)
    confidence_df.to_csv(os.path.join(output_dir, "confidence_analysis.csv"), index=False)
    explanation_df.to_csv(os.path.join(output_dir, "explanation_analysis.csv"), index=False)
    runtime_df.to_csv(os.path.join(output_dir, "runtime_profile.csv"), index=False)
    grounding_df.to_csv(os.path.join(output_dir, "agent_grounding_analysis.csv"), index=False)
    evidence_df.to_csv(os.path.join(output_dir, "evidence_provenance_analysis.csv"), index=False)
    errors.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)
    invalid.to_csv(os.path.join(output_dir, "invalid_decisions.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("FIGDEBATE EVALUATION SUMMARY\n" + "=" * 40 + "\n\n")
        handle.write(f"Samples: {total}\nValid Predictions: {len(valid)}\nInvalid Predictions: {len(invalid)}\n")
        handle.write(f"Accuracy: {accuracy:.4f}\nBalanced Accuracy: {balanced_accuracy:.4f}\nMacro F1: {macro_f1:.4f}\n\n")
        for label in LABELS:
            values = per_label[label]
            handle.write(f"{label}\nPrecision: {values['precision']:.4f}\nRecall: {values['recall']:.4f}\nF1: {values['f1']:.4f}\n\n")
        reliability = metrics["decision_reliability"]
        handle.write(f"Primary Arbiter Valid Rate: {reliability['primary_arbiter_valid_rate']}\n")
        handle.write(f"Format Retry Rate: {reliability['format_retry_rate']}\n")
        handle.write(f"Citation Retry Rate: {reliability['citation_retry_rate']}\n")
        handle.write(f"Binary Resolution Rate: {reliability['binary_resolution_rate']}\n")
        handle.write(f"Forced Label Rate: {reliability['forced_label_rate']}\n")
        handle.write(f"Decision Method Distribution: {reliability['decision_method_distribution']}\n\n")
        handle.write(
            "Review Board Status Distribution: "
            f"{reliability['review_board_status_distribution']}\n"
        )
        handle.write(
            "Review Board Directionally Grounded Rate: "
            f"{reliability['review_board_directionally_grounded_rate']}\n\n"
        )
        handle.write(
            "Review Board Confidence Cap Rate: "
            f"{reliability['review_board_confidence_cap_rate']}\n\n"
        )
        handle.write(f"Prediction Distribution: {prediction_distribution}\n")
        handle.write(f"ENTAILS Prediction Rate: {entails_prediction_rate:.4f}\n")
        handle.write(
            "Comparator Evidence Distribution: "
            f"{metrics['comparator']['evidence_status_distribution']}\n"
        )
        handle.write(
            "Comparator Text Surfaces Without OCR: "
            f"{metrics['comparator']['text_surface_without_ocr_count']}\n"
        )
        handle.write(
            f"Agent 1 Schema Complete Rate: {metrics['grounding']['schema_complete_rate']:.4f}\n\n"
        )
        handle.write(
            f"Agent 1 Format Valid Rate: {metrics['grounding']['format_valid_rate']}\n"
            f"Agent 1 Factual Grounding Rate: {metrics['grounding']['factual_grounding_rate']}\n"
            f"Agent 1 OCR Usable Rate: {metrics['grounding']['ocr_usable_rate']}\n"
            f"Agent 1 Relation Binding Rate: {metrics['grounding']['relation_binding_rate']}\n"
            f"Agent 1 Targeted Recovery Attempts: {metrics['grounding']['targeted_recovery_attempt_count']}\n"
            f"Agent 1 Targeted Recovery Success Rate: {metrics['grounding']['targeted_recovery_success_rate']}\n\n"
        )
        handle.write(
            f"Agent 1 Schema Retry Rate: {metrics['grounding']['schema_retry_rate']}\n"
        )
        handle.write(
            "Agent 1 Schema Retry Success Rate: "
            f"{metrics['grounding']['schema_retry_success_rate']}\n\n"
        )
        handle.write(
            "Agent 1 Symbolic Evidence Rate: "
            f"{metrics['grounding']['symbolic_evidence_rate']}\n"
            "Mean Visual Metaphor Items: "
            f"{metrics['grounding']['mean_visual_metaphor_items']}\n\n"
        )
        handle.write(
            "Caption-Type/Phenomenon Agreement (diagnostic only): "
            f"{type_agreement}\n"
        )
        handle.write(
            "Note: the dataset phenomenon may occur in the image, caption, or both; "
            "this is not a caption-classification accuracy metric.\n"
        )
        handle.write(
            f"Structured Claim Relation Resolved Rate: {metrics['claim_relations']['resolved_rate']}\n"
            f"Immutable Claim Contract Valid Rate: {metrics['claim_relations']['contract_valid_rate']}\n"
            f"Claim Proposition Preserved Rate: {metrics['claim_relations']['proposition_preserved_rate']}\n"
            f"Claim Entity Frame Preserved Rate: {metrics['claim_relations']['entity_frame_preserved_rate']}\n"
            f"Structured Claim Retry Rate: {metrics['claim_relations']['retry_rate']}\n"
            f"Structured Claim Retry Success Rate: {metrics['claim_relations']['retry_success_rate']}\n"
            f"Structured Relation Candidates: {metrics['claim_relations']['structured_candidate_count']}\n\n"
        )
        handle.write(
            "Explanation Token F1: "
            f"{metrics['explanations']['mean_token_f1']}\n"
            "Explanation ROUGE-L F1: "
            f"{metrics['explanations']['mean_rouge_l_f1']}\n\n"
        )
        handle.write(f"Confidence Mean: {confidence['mean']}\nBrier Score: {confidence['brier_score']}\nECE (10 bins): {confidence['ece_10_bin']}\n\n")
        handle.write(
            "Decision-Grade Directional Evidence Rate: "
            f"{metrics['evidence_provenance']['direct_evidence_valid_rate']}\n"
        )
        handle.write(
            "Grounded Source Citation Rate: "
            f"{metrics['evidence_provenance']['grounded_source_citation_rate']}\n"
        )
        handle.write(
            "Evidence Status Distribution: "
            f"{metrics['evidence_provenance']['final_status_distribution']}\n\n"
        )
        handle.write(
            "Generic NLI Decision Promotions: "
            f"{metrics['evidence_provenance']['atomic_verified_count']}/"
            f"{metrics['evidence_provenance']['atomic_candidate_count']} "
            f"({metrics['evidence_provenance']['atomic_verification_rate']})\n\n"
        )
        handle.write(f"Debate Trigger Count: {metrics['debate']['trigger_count']}\nDebate Trigger Rate: {metrics['debate']['trigger_rate']:.4f}\nDebate Level Distribution: {metrics['debate']['level_distribution']}\nDebate Mean Need Score: {metrics['debate']['mean_need_score']}\nDebate Revision Acceptance Rate: {metrics['debate']['revision_acceptance_rate']}\nAgent 1 Critique Format Valid Rate: {metrics['debate']['visual_critique_format_valid_rate']}\nAgent 1 Critique Retry Count: {metrics['debate']['visual_critique_retry_count']}\nAgent 1 Critique Retry Success Rate: {metrics['debate']['visual_critique_retry_success_rate']}\nVisual Evidence Consensus Proposals: {metrics['debate']['visual_evidence_consensus_count']}\nAgent 2 Critique Format Valid Rate: {metrics['debate']['linguistic_critique_format_valid_rate']}\nDebate Review Status Distribution: {metrics['debate']['review_status_distribution']}\nDebate Correction Rate: {correction_rate:.4f}\nDebate Harm Rate: {harm_rate:.4f}\n")
        handle.write(f"Debate Corrections: {debate_corrections}\nDebate Harms: {debate_harms}\nDebate Net Correct Decisions: {debate_net_gain}\n")
        handle.write(
            "Decision-Grade Visual Reinspection Evidence: "
            f"{metrics['debate']['visual_reinspection_evidence_count']} "
            "items across "
            f"{metrics['debate']['visual_reinspection_evidence_case_count']} cases\n"
        )
        handle.write(f"Region OCR Reviews: {metrics['debate']['region_ocr_review_count']}\nTargeted Region Verifier Attempts: {metrics['debate']['targeted_region_verifier_attempt_count']}\nTargeted Region Verifier Accepted: {metrics['debate']['targeted_region_verifier_accepted_count']}\nTargeted Region Proposal Accuracy: {metrics['debate']['targeted_region_verifier_proposal_accuracy']}\nTargeted Region Accepted Accuracy: {metrics['debate']['targeted_region_verifier_accuracy']}\n")
        handle.write(f"Feedback Role Distribution: {metrics['feedback']['role_distribution']}\nFeedback Matched Samples: {metrics['feedback']['memory_active_samples']}\nFeedback Match Rate: {metrics['feedback']['memory_match_rate']}\nFeedback Matched Rule Distribution: {metrics['feedback']['matched_rule_distribution']}\nFeedback Revision Acceptances: {metrics['feedback']['revision_acceptance_count']}\nFeedback Corrections: {metrics['feedback']['correction_count']}\nFeedback Harms: {metrics['feedback']['harm_count']}\nFeedback Net Correct Decisions: {metrics['feedback']['net_correct_decisions']}\nFeedback Candidates: {metrics['feedback']['candidate_count']}\nFeedback Updates: {metrics['feedback']['update_count']}\n\n")
        handle.write(f"Judge/Mediator Requested: {metrics['judge']['requested_count']}\nJudge/Mediator Contract Valid Rate: {metrics['judge']['contract_valid_rate']}\nJudge/Mediator Verdict Distribution: {metrics['judge']['verdict_distribution']}\nMediator Usable Rate: {metrics['judge']['mediator_usable_rate']}\nMediated Tie-breaks: {metrics['judge']['mediated_tiebreak_count']}\nJudge Counterfactual Corrections: {metrics['judge']['counterfactual_corrections']}\nJudge Counterfactual Harms: {metrics['judge']['counterfactual_harms']}\nJudge Counterfactual Net Correct: {metrics['judge']['counterfactual_net_correct']}\nJudge Accepted Revisions: {metrics['judge']['accepted_revisions']}\nJudge Accepted Corrections: {metrics['judge']['accepted_corrections']}\nJudge Accepted Harms: {metrics['judge']['accepted_harms']}\nJudge Accepted Net Correct: {metrics['judge']['accepted_net_correct']}\nJudge Feedback Review Candidates: {metrics['judge']['feedback_review_candidates']}\nJudge Feedback Memory Updates: {metrics['judge']['feedback_memory_updates']}\n\n")
        handle.write("Paper artifacts: confusion_matrix.csv, phenomenon_breakdown.csv, decision_method_breakdown.csv, comparator_analysis.csv, debate_analysis.csv, debate_log.csv, debate_log.jsonl, feedback_analysis.csv, feedback_decision_log.csv, feedback_decision_log.jsonl, judge_analysis.csv, evidence_provenance_analysis.csv, confidence_analysis.csv, explanation_analysis.csv, runtime_profile.csv, agent_grounding_analysis.csv, error_analysis.csv.\n")
    print(f"Saved metrics and paper artifacts to {output_dir}")
    return metrics


def main():
    args = parse_args()
    evaluate_predictions(args.input, args.output_dir)


if __name__ == "__main__":
    main()
