"""Compare every locked local row to hash-verified, pinned Hub parquet files.

Writes a provenance sidecar only. It never changes a label, pixel or split.
Exact reproduction of the historical JPEG85 conversion is accepted as lineage,
not as proof that lossy conversion or the native annotation is semantically sound.
"""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import pickle

from PIL import Image
from dataset.prepare_vflute import normalize_label
from dataset.protocol import DATASET_ID, DATASET_REVISION, included, image_identity


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare_row(local, source, cached=None):
    values = {
        "caption": source["claim"], "label": normalize_label(source["label"]),
        "phenomenon": source["phenomenon"], "explanation": source["explanation"],
    }
    mismatches, transforms = [], {}
    for key, value in values.items():
        if local.get(key) == value:
            continue
        if key in {"caption", "explanation"} and isinstance(value, str) and local.get(key) == value.strip():
            transforms[key] = "OUTER_WHITESPACE_STRIPPED"
        else:
            mismatches.append(key)
    original = source["image"]["bytes"]
    if cached is not None:
        if (cached["local_image_sha256"] != hashlib.sha256(local["image_bytes"]).hexdigest()
                or cached["upstream_image_sha256"] != hashlib.sha256(original).hexdigest()):
            raise ValueError("Cached image verification no longer matches these bytes")
        result = dict(cached)
        result.update(field_mismatches=mismatches, text_transformations=transforms,
                      lineage_verified=not mismatches and cached["image_lineage"] != "UNVERIFIED_TRANSFORMATION")
        return result
    identity = image_identity(original)
    legacy = image_identity(local["image_bytes"])
    status = "UNVERIFIED_TRANSFORMATION"
    if identity["image_sha256"] == legacy["image_sha256"]:
        status = "EXACT_BYTES"
    elif identity["image_group_id"] == legacy["image_group_id"]:
        status = "EXACT_DECODED_PIXELS"
    else:
        with Image.open(BytesIO(original)) as image:
            output = BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=85)
        if output.getvalue() == local["image_bytes"]:
            status = "EXACT_REPRODUCED_JPEG85"
    return {"id": local["id"], "field_mismatches": mismatches, "text_transformations": transforms,
            "image_lineage": status, "source_dataset": source["source_dataset"],
            "native_label": source["label"], "upstream_dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION, "local_image_sha256": legacy["image_sha256"],
            "upstream_image_sha256": identity["image_sha256"],
            "upstream_image_group_id": identity["image_group_id"],
            "caption_sha256": hashlib.sha256(local["caption"].encode()).hexdigest(),
            "label": local["label"], "phenomenon": local["phenomenon"],
            "explanation_sha256": hashlib.sha256(local.get("explanation", "").encode()).hexdigest(),
            "lineage_verified": not mismatches and status != "UNVERIFIED_TRANSFORMATION"}


def verify(upstream_dir, processed_dir, output_dir, reuse_image_verification=None):
    import pyarrow.parquet as pq
    upstream, processed, output = map(Path, (upstream_dir, processed_dir, output_dir))
    output.mkdir(parents=True, exist_ok=False)
    files = sorted((upstream / "data").glob("*.parquet"))
    if len(files) != 12:
        raise ValueError("Expected all twelve pinned parquet shards")
    inventory = []
    for path in files:
        metadata = upstream / ".cache/huggingface/download/data" / (path.name + ".metadata")
        revision, etag, *_ = metadata.read_text().splitlines()
        digest = file_sha(path)
        if revision != DATASET_REVISION or etag != digest:
            raise ValueError(f"Pinned upstream file identity failed: {path.name}")
        inventory.append({"file": path.name, "sha256": digest, "bytes": path.stat().st_size})
    cache, previous = {}, None
    if reuse_image_verification:
        previous_dir = Path(reuse_image_verification)
        previous = json.loads((previous_dir / "verification.json").read_text())
        if previous["revision"] != DATASET_REVISION or previous["upstream_files"] != inventory:
            raise ValueError("Previous verification uses different upstream files")
        cache = {r["id"]: r for r in map(json.loads, (previous_dir / "provenance.jsonl").read_text().splitlines())}
    reports, split_summaries = [], []
    for split, local_name in (("train", "vflute_train"), ("validation", "vflute_val"), ("test", "vflute_test")):
        path = processed / (local_name + ".pkl")
        before = file_sha(path)
        if previous and before != next(s["local_file_sha256"] for s in previous["splits"] if s["split"] == local_name):
            raise ValueError("Local data changed since previous image verification")
        with path.open("rb") as handle:
            records = pickle.load(handle)
        local = {row["id"]: row for row in records}
        if len(local) != len(records):
            raise ValueError("Duplicate local IDs")
        index, selected, seen = 0, 0, set()
        start = len(reports)
        for shard in sorted((upstream / "data").glob(split + "-*.parquet")):
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=8):
                for source in batch.to_pylist():
                    sample_id = f"vflute_{split}_{index}"
                    index += 1
                    if not included(source):
                        continue
                    selected += 1
                    if sample_id not in local:
                        reports.append({"id": sample_id, "lineage_verified": False,
                                        "error": "MISSING_LOCAL_ROW", "upstream_split": split})
                        continue
                    result = compare_row(local[sample_id], source, cache.get(sample_id))
                    result.update(upstream_split=split, upstream_row_index=index - 1)
                    reports.append(result)
                    seen.add(sample_id)
        extras = sorted(local.keys() - seen)
        split_summaries.append({"split": local_name, "upstream_total": index,
            "selected_upstream": selected, "local_rows": len(local), "extra_local_ids": extras,
            "local_file_sha256": before, "local_unchanged": before == file_sha(path),
            "verified_rows": sum(row["lineage_verified"] for row in reports[start:]),
            "native_source_counts": dict(Counter(row.get("source_dataset", "UNKNOWN") for row in reports[start:]))})
        print(json.dumps(split_summaries[-1]), flush=True)
    # The development subset is a view of train, not an additional partition.
    by_id = {row["id"]: row for row in reports}
    dev_path = processed / "vflute_train_dev50.pkl"
    with dev_path.open("rb") as handle:
        dev = pickle.load(handle)
    dev_matches = len(dev) == 50 and len({row["id"] for row in dev}) == 50 and all(row["id"] in by_id and by_id[row["id"]].get("lineage_verified")
        and hashlib.sha256(row["image_bytes"]).hexdigest() == by_id[row["id"]]["local_image_sha256"]
        and hashlib.sha256(row["caption"].encode()).hexdigest() == by_id[row["id"]]["caption_sha256"]
        and row["label"] == by_id[row["id"]]["label"]
        and row["phenomenon"] == by_id[row["id"]]["phenomenon"]
        and hashlib.sha256(row.get("explanation", "").encode()).hexdigest() == by_id[row["id"]]["explanation_sha256"] for row in dev)
    summary = {"dataset_id": DATASET_ID, "revision": DATASET_REVISION,
        "upstream_files": inventory, "splits": split_summaries,
        "dev50_rows": len(dev), "dev50_matches_verified_train": dev_matches,
        "dev50_file_sha256": file_sha(dev_path),
        "image_lineage_counts": dict(Counter(row.get("image_lineage", "MISSING") for row in reports)),
        "text_transformation_counts": dict(Counter(key for row in reports for key in row.get("text_transformations", {}))),
        "all_source_lineage_verified": all(row["lineage_verified"] for row in reports) and dev_matches
            and all(not s["extra_local_ids"] and s["local_unchanged"] for s in split_summaries),
        "publication_qualified": False,
        "remaining_gates": ["independent semantic annotation and adjudication", "review near-duplicate candidates",
            "freeze contamination-aware evaluation protocol", "image readability review"],
        "legacy_task_status": {"capcon": "QUARANTINED_UNVERIFIED_NEGATIVE_CONSTRUCTION",
                               "mmsd2": "QUARANTINED_NATIVE_SARCASM_NOT_VISUAL_ENTAILMENT"}}
    (output / "provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reports), encoding="utf8")
    summary["provenance_sha256"] = file_sha(output / "provenance.jsonl")
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-image-verification", help="Prior local verification; all file and per-row byte identities are rechecked")
    args = parser.parse_args()
    result = verify(args.upstream_dir, args.processed_dir, args.output_dir, args.reuse_image_verification)
    print(json.dumps({"all_source_lineage_verified": result["all_source_lineage_verified"],
                      "image_lineage_counts": result["image_lineage_counts"]}), flush=True)
    raise SystemExit(0 if result["all_source_lineage_verified"] else 2)
