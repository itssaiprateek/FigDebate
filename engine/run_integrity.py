"""Run-directory integrity checks without model-runtime dependencies."""

import json
import os


IMMUTABLE_RUN_KEYS = (
    "batch_size", "source_manifest_sha256",
    "data_usage", "evaluation_reference_sha256",
    "candidate_mode", "control_mode", "selection_seed", "stage_fingerprints",
    "dataset", "requested_samples", "execution_mode", "debate_mode",
    "evidence_mode", "judge_mode", "judge_scope", "feedback_mode",
    "semantic_bridge_mode", "hardware_profile", "ablation_signature",
    "verified_feedback_sha256",
    "model_vision", "model_vision_revision", "model_language",
    "model_language_revision", "seed", "selection_strategy",
    "model_judge", "model_judge_revision",
    "dataset_selection_sha256",
    "selection_manifest_sha256", "selection_manifest_schema",
    "pipeline_source_sha256", "evidence_ledger_version",
)


def validate_resume_config(run_dir, requested):
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        raise ValueError("Cannot resume: run_config.json is missing.")
    with open(path, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    mismatches = {
        key: {"existing": existing.get(key), "requested": requested.get(key)}
        for key in IMMUTABLE_RUN_KEYS
        if existing.get(key) != requested.get(key)
    }
    if mismatches:
        details = "; ".join(
            f"{key}: {value['existing']!r} != {value['requested']!r}"
            for key, value in mismatches.items()
        )
        raise ValueError(f"Cannot resume with incompatible configuration: {details}")
    return existing
