"""Select within one official split; never move rows between splits."""
import json
from pathlib import Path
from engine.sampling import build_selection_manifest, select_from_manifest


def sample_count(value):
    if str(value).lower() == "all":
        return None
    count = int(value)
    if count < 1:
        raise ValueError("Sample count must be positive, or 'all'")
    return count


def select_run(records, split, count=1, strategy=None, selection_seed=None, inference_seed=42,
               manifest_path=None, sample_ids=None, ids_file=None, phenomena=None, sources=None):
    if manifest_path:
        if any((sample_ids, ids_file, phenomena, sources)):
            raise ValueError("Replay a manifest OR define a new cohort, not both")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf8"))
        if manifest.get("dataset_split", split) != split:
            raise ValueError("Manifest belongs to a different official split")
        if strategy is not None and strategy != manifest["strategy"]:
            raise ValueError("Replay strategy differs from manifest")
        if selection_seed is not None and selection_seed != manifest["seed"]:
            raise ValueError("Replay selection seed differs from manifest")
    else:
        if sample_ids and ids_file:
            raise ValueError("Specify IDs directly or in a file, not both")
        identifiers = sample_ids
        if ids_file:
            identifiers = json.loads(Path(ids_file).read_text(encoding="utf8"))
        if identifiers is not None:
            if not isinstance(identifiers, list) or not identifiers or any(not isinstance(k, str) or not k.strip() for k in identifiers):
                raise ValueError("Sample IDs must be a nonempty JSON list of strings")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Duplicate requested IDs")
        pool = list(records)
        by_id = {r["id"]: r for r in pool}
        if len(by_id) != len(pool):
            raise ValueError("Duplicate source IDs")
        if identifiers is not None:
            missing = set(identifiers) - by_id.keys()
            if missing:
                raise ValueError("IDs not in the selected split: " + ", ".join(sorted(missing)))
            pool = [by_id[key] for key in identifiers]
        for field, requested in (("phenomenon", phenomena), ("source_dataset", sources)):
            if requested:
                unknown = set(requested) - {r.get(field) for r in records}
                if unknown:
                    raise ValueError(f"Unknown {field} filter: {sorted(unknown)}")
                pool = [r for r in pool if r.get(field) in requested]
        if not pool:
            raise ValueError("The requested cohort is empty")
        strategy = strategy or ("prefix" if identifiers is not None else "random")
        seed = inference_seed if selection_seed is None else selection_seed
        manifest = build_selection_manifest(pool, strategy=strategy, seed=seed,
            dataset_split=split, cohort={"ids": identifiers, "phenomena": phenomena, "sources": sources})
    actual_count = len(manifest["records"]) if count is None else count
    selected = select_from_manifest(records, manifest, actual_count)
    return selected, manifest


def usage_policy(split, purpose, feedback_mode):
    if purpose not in {"diagnostic", "evaluation"}:
        raise ValueError("Unknown run purpose")
    if feedback_mode in {"collect", "calibrate"} and (split == "vflute_test" or purpose != "diagnostic"):
        raise ValueError("Gold-informed feedback collection/calibration requires a diagnostic train or validation run")
    if purpose == "evaluation" and split not in {"vflute_val", "vflute_test"}:
        raise ValueError("Train/dev50 are development data; use diagnostic purpose")
    return {"purpose": purpose, "official_split": split, "split_membership_changed": False,
            "reference_explanations": "post_prediction_scoring_only",
            "gold_feedback_updates_permitted": purpose == "diagnostic" and split != "vflute_test",
            "untouched_test_status": "not_established_by_a_command_line_flag",
            "note": "Repeated test cases may be inspected diagnostically, but once used to tune the system they are not untouched final evidence."}
