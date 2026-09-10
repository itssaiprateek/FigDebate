"""Fail-closed configuration pairing for treatment comparisons."""
import json
from pathlib import Path

TREATMENT_KEYS = frozenset({
    "debate_mode", "judge_mode", "judge_scope", "semantic_bridge_mode",
    "candidate_mode", "evidence_mode", "control_mode",
    "feedback_mode", "verified_feedback_sha256",
})
INVARIANT_KEYS = (
    "data_usage", "evaluation_reference_sha256",
    "dataset", "requested_samples", "selection_seed", "selection_strategy",
    "selection_manifest_sha256", "dataset_selection_sha256", "seed",
    "model_vision", "model_vision_revision", "model_language", "model_language_revision",
    "model_judge", "model_judge_revision", "pipeline_source_sha256",
    "hardware_profile", "feedback_mode", "verified_feedback_sha256",
    "runtime_environment", "deterministic_environment", "evidence_ledger_version",
)


def verify_configuration(control_path, treatment_path, allowed_changes=(), allow_legacy=False, code_change_description=None):
    paths = [Path(path).parent / "run_config.json" for path in (control_path, treatment_path)]
    if not all(path.is_file() for path in paths):
        if allow_legacy:
            return {"verified": False, "reason": "missing_run_configuration"}
        raise ValueError("Both runs require run_config.json; legacy comparisons are diagnostic only")
    left, right = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    allowed = set(allowed_changes)
    if allowed - TREATMENT_KEYS:
        raise ValueError("Only declared treatment settings may differ; inputs, models and code must match")
    missing = [key for key in INVARIANT_KEYS if key not in left or key not in right]
    if missing and not allow_legacy:
        raise ValueError("Run configuration missing required identities: " + ", ".join(missing))
    differences = {key: [left.get(key), right.get(key)] for key in INVARIANT_KEYS
                   if left.get(key) != right.get(key) and key not in allowed}
    code_change = None
    if code_change_description is not None:
        if not code_change_description.strip() or "pipeline_source_sha256" not in differences or not all(
                isinstance(config.get("pipeline_source_sha256"), str) and config["pipeline_source_sha256"] for config in (left, right)):
            raise ValueError("A declared code comparison requires a description and two different nonempty pipeline hashes")
        code_change = {"description": code_change_description.strip(),
                       "pipeline_hashes": differences.pop("pipeline_source_sha256"),
                       "scope": "whole_declared_code_revision_not_proof_of_one_isolated_change"}
    if differences:
        raise ValueError("Confounded comparison; invariant settings differ: " + ", ".join(differences))
    treatments = {key: [left.get(key), right.get(key)] for key in TREATMENT_KEYS
                  if left.get(key) != right.get(key)}
    if set(treatments) != allowed:
        raise ValueError("Declared treatment differences must exactly match observed differences: "
                         + ", ".join(sorted(treatments)))
    return {"verified": not missing, "declared_treatment_differences": treatments,
            "declared_code_change": code_change,
            "missing_legacy_keys": missing}
