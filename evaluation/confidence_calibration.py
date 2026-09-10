"""Fit a separate reported-correctness calibrator; never relax evidence gates.

The sigmoid method follows standard post-hoc score calibration. This is not
temperature scaling of model logits. Actual utility must be measured out of sample.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

METHOD_KEYS = ("pipeline_source_sha256", "ablation_signature", "hardware_profile",
               "model_vision_revision", "model_language_revision", "model_judge_revision",
               "feedback_mode", "verified_feedback_sha256", "runtime_environment")


def method_identity(config):
    missing = [key for key in METHOD_KEYS if key not in config]
    if missing:
        raise ValueError("Missing method identities: " + ", ".join(missing))
    return {key: config[key] for key in METHOD_KEYS}


def probability(score, artifact):
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Uncalibrated score must be finite and between zero and one")
    z = artifact["coefficient"] * score + artifact["intercept"]
    return 1 / (1 + math.exp(-max(-700, min(700, z))))


def fit(records, config, audit):
    if config.get("dataset") != "vflute_train":
        raise ValueError("Fit only on the locked internal calibration partition")
    partition = audit["internal_train_partition"]
    if "excluded_previously_used_or_external_groups" not in partition:
        raise ValueError("Audit lacks inspected-image exclusions")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Calibration requires unique, nonempty case IDs")
    if not set(ids) <= set(partition["calibration_ids"]):
        raise ValueError("Calibration includes cases outside its locked partition")
    expected = {row["id"]: row for split in audit["splits"] if split["split"] == "vflute_train"
                for row in split["samples"]}
    for row in records:
        if row.get("ground_truth") != expected[row["id"]]["label"]:
            raise ValueError("Calibration gold label does not match the audited dataset")
        for key in ("image_sha256", "caption_sha256", "image_group_id"):
            if not row.get(key) or row[key] != expected[row["id"]][key]:
                raise ValueError("Calibration input identity mismatch")
    groups = {row["image_group_id"] for row in records}
    if groups & set(partition["excluded_previously_used_or_external_groups"]):
        raise ValueError("Calibration overlaps inspected or external image groups")
    if len(groups) < 20:
        raise ValueError("Fewer than 20 image groups: insufficient for even an exploratory fit")
    scores = [float(row["final_confidence"]) for row in records]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError("Invalid confidence score")
    if any(row["prediction"] not in {"ENTAILS", "CONTRADICTS"} for row in records):
        raise ValueError("Calibrate delivered binary answers, not missing outputs")
    targets = [int(row["prediction"] == row["ground_truth"]) for row in records]
    if len(set(targets)) != 2 or len(set(scores)) < 2:
        raise ValueError("Need both correct/incorrect answers and variable scores")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    model.fit([[score] for score in scores], targets)
    return {"schema_version": "1.0", "method": "regularized_sigmoid_of_reported_score",
            "coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]),
            "C": 1.0, "target": "delivered_answer_correctness", "method_identity": method_identity(config),
            "calibration_ids": ids, "calibration_image_groups": sorted(groups),
            "samples": len(records), "groups": len(groups), "fit_is_not_validation": True,
            "acceptance_thresholds_changed": False, "human_faithfulness_evaluated": False}


def apply(records, config, artifact):
    if method_identity(config) != artifact["method_identity"]:
        raise ValueError("Frozen calibration does not match this method/configuration")
    calibration_groups = set(artifact["calibration_image_groups"])
    if any(not row.get("image_group_id") or row["image_group_id"] in calibration_groups for row in records):
        raise ValueError("Calibrator must be evaluated on disjoint known image groups")
    return [{"id": row["id"], "uncalibrated_score": row["final_confidence"],
             "calibrated_correctness_probability": probability(row["final_confidence"], artifact),
             "prediction": row["prediction"], "prediction_changed": False} for row in records]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "apply"))
    parser.add_argument("--records", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit")
    parser.add_argument("--calibrator")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Calibration artifacts are immutable")
    raw = Path(args.records).read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "fit":
        if not args.audit:
            parser.error("fit requires --audit")
        audit_bytes = Path(args.audit).read_bytes()
        result = fit(records, config, json.loads(audit_bytes))
        result.update(records_sha256=hashlib.sha256(raw).hexdigest(),
                      audit_sha256=hashlib.sha256(audit_bytes).hexdigest())
    else:
        if not args.calibrator:
            parser.error("apply requires --calibrator")
        result = apply(records, config, json.loads(Path(args.calibrator).read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
