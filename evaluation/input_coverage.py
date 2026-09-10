"""Reconcile delivered outputs with the intended, signed sample manifest."""
import hashlib
import json
from pathlib import Path
import pandas as pd


def reconcile(frame, input_path):
    if "id" in frame and (frame["id"].isna().any() or frame["id"].astype(str).str.strip().eq("").any() or frame["id"].astype(str).duplicated().any()):
        raise ValueError("Evaluation IDs must be nonempty and unique")
    parent = Path(input_path).resolve().parent
    manifest_path, config_path = parent / "sample_manifest.json", parent / "run_config.json"
    if not manifest_path.exists() or not config_path.exists():
        return frame, {"status": "UNVERIFIED_NO_INTENDED_MANIFEST", "delivered_rows": len(frame),
                       "intended_rows": None, "missing_ids": [], "publication_qualified": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    config = json.loads(config_path.read_text(encoding="utf8"))
    if "dataset_split" in manifest and manifest["dataset_split"] != config.get("dataset"):
        raise ValueError("Run configuration and manifest name different official splits")
    unsigned = dict(manifest)
    declared = unsigned.pop("manifest_sha256", None)
    observed = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if declared != observed or config.get("sample_manifest_sha256", config.get("selection_manifest_sha256")) != declared:
        raise ValueError("Intended selection manifest identity mismatch")
    count = config.get("requested_samples")
    entries = manifest.get("records", [])
    if type(count) is not int or not 0 < count <= len(entries):
        raise ValueError("Invalid intended sample count")
    expected = {str(r["id"]): r for r in entries[:count]}
    if len(expected) != count or "id" not in frame:
        raise ValueError("Intended/delivered IDs are invalid")
    from evaluation.reference_bank import load_references, text_hash
    references = load_references(parent, manifest, expected)
    delivered = set(frame["id"].astype(str))
    if delivered - expected.keys():
        raise ValueError("Delivered records outside the intended selection")
    for row in frame.to_dict("records"):
        source = expected[str(row["id"])]
        for field, source_key in (("ground_truth", "label"), ("phenomenon", "phenomenon"),
                                  ("image_sha256", "image_sha256"), ("caption_sha256", "caption_sha256")):
            if row.get(field) != source.get(source_key) or not source.get(source_key):
                raise ValueError(f"Evaluation identity mismatch: {row['id']} / {field}")
        if references is not None:
            value = row.get("reference_explanation")
            if value is not None and not pd.isna(value) and str(value).strip():
                if text_hash(str(value)) != references[str(row["id"])]["reference_explanation_sha256"]:
                    raise ValueError("Prediction file contains a different reference explanation")
    missing = [key for key in expected if key not in delivered]
    placeholders = [dict(id=key, ground_truth=expected[key]["label"],
                         phenomenon=expected[key]["phenomenon"], prediction=None,
                         image_sha256=expected[key].get("image_sha256"),
                         caption_sha256=expected[key].get("caption_sha256"),
                         image_group_id=expected[key].get("image_group_id"),
                         final_decision_valid=False, evaluation_execution_status="MISSING_OUTPUT") for key in missing]
    result = pd.concat([frame, pd.DataFrame(placeholders)], ignore_index=True) if placeholders else frame
    result = result.copy()
    if references is not None:
        result["reference_explanation"] = result["id"].astype(str).map(
            lambda key: references[key]["reference_explanation"])
    result["verified_source_dataset"] = result["id"].astype(str).map(
        lambda key: expected[key].get("source_dataset", "UNKNOWN_LEGACY"))
    return result, {"status": "RECONCILED", "delivered_rows": len(frame), "intended_rows": count,
                    "missing_ids": missing, "coverage": len(frame) / count,
                    "missing_outputs_count_as_errors": True, "publication_qualified": False,
                    "source_lineage_verified": all(row.get("source_lineage_verified") is True for row in expected.values()),
                    "reference_bank_available": references is not None,
                    "references_bound_to_manifest": references is not None and all("reference_explanation_sha256" in r for r in expected.values()),
                    "data_usage": config.get("data_usage", {"purpose": "legacy_unspecified", "untouched_test_status": "not_established"}),
                    "selection_strategy": manifest.get("strategy"),
                    "selection_role": "diagnostic_balanced_subset_not_natural_prevalence" if manifest.get("strategy") == "stratified" else "declared_selection",
                    "note": "Input consistency is not dataset or semantic qualification."}
