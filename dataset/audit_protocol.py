"""Audit trusted local split files; never rewrite, relabel or delete samples."""
import argparse
import json
from pathlib import Path
import pickle
from dataset.protocol import manifest, grouped_partition, cross_split_overlaps
from dataset.loaders import attach_provenance


def audit(paths, output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("Choose a new audit directory; existing audits are immutable")
    manifests = []
    for path in paths:
        # Only load trusted local project pickles, not files from untrusted users.
        with Path(path).open("rb") as handle:
            records = pickle.load(handle)
        records = attach_provenance(records, Path(path).stem, path)
        manifests.append(manifest(records, Path(path).stem))
    report = {"splits": manifests, "overlaps": cross_split_overlaps(manifests),
              "dataset_mutated": False, "near_duplicates": [], "strict_test_supplement_ids": [],
              "publication_qualified": False,
              "strict_test_scope": "exact_pixel_disjoint_only; near duplicates and semantic annotations remain unqualified",
              "partition_role": "internal exploratory development/calibration; not a new untouched test set"}
    training = next((m for m in manifests if m["split"] == "vflute_train"), None)
    if training:
        used_groups = {row["image_group_id"] for m in manifests
                       if m["split"] != "vflute_train" for row in m["samples"]}
        fresh = [row for row in training["samples"] if row["image_group_id"] not in used_groups]
        report["internal_train_partition"] = grouped_partition(fresh)
        report["internal_train_partition"]["excluded_previously_used_or_external_groups"] = sorted(used_groups)
        report["internal_train_partition"]["eligible_samples"] = len(fresh)
    development_groups = {row["image_group_id"] for m in manifests
                          if "test" not in m["split"] for row in m["samples"]}
    report["strict_test_supplement_ids"] = [
        row["id"] for m in manifests if m["split"] == "vflute_test"
        for row in m["samples"] if row["image_group_id"] not in development_groups]
    # Candidate list only: dHash is not proof of duplication and never removes rows.
    unique = {}
    for m in manifests:
        for row in m["samples"]:
            unique.setdefault(row["image_group_id"], (row["id"], row["image_dhash"]))
    values = list(unique.values())
    for i, (left, left_hash) in enumerate(values):
        for right, right_hash in values[i + 1:]:
            distance = (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()
            if distance <= 2:
                report["near_duplicates"].append(
                    {"left": left, "right": right, "dhash_distance": distance, "human_review_required": True})
    target.mkdir(parents=True)
    (target / "protocol_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", required=True, help="Trusted local pickle paths")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.splits, args.output)
    print(json.dumps({"counts": {m["split"]: m["rows"] for m in result["splits"]},
                      "overlap_group_counts": [{"left": x["left"], "right": x["right"],
                        "groups": len(x["shared_pixel_groups"])} for x in result["overlaps"]],
                      "near_duplicate_candidates": len(result["near_duplicates"]),
                      "dataset_mutated": False}, indent=2))
