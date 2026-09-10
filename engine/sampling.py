"""Deterministic dataset sampling policies for comparable experiments."""

from collections import defaultdict
import hashlib
import json
import random


def select_records(records, count, strategy="stratified", seed=42):
    records = list(records or [])
    count = max(0, min(int(count), len(records)))
    if strategy == "prefix":
        return records[:count]
    rng = random.Random(seed)
    if strategy == "random":
        indices = list(range(len(records)))
        rng.shuffle(indices)
        return [records[index] for index in indices[:count]]
    if strategy != "stratified":
        raise ValueError(f"Unknown selection strategy: {strategy}")

    groups = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            str(record.get("phenomenon", "unknown")),
            str(record.get("label", "unknown")),
        )
        groups[key].append((index, record))
    for rows in groups.values():
        rng.shuffle(rows)

    selected = []
    ordered_keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return [record for _, record in selected]


def record_manifest_entry(record):
    image_bytes = record.get("image_bytes") or b""
    if isinstance(image_bytes, str):
        image_bytes = image_bytes.encode("utf-8")
    elif not isinstance(image_bytes, bytes):
        image_bytes = bytes(image_bytes)
    result = {
        "id": str(record.get("id") or ""),
        "label": str(record.get("label") or ""),
        "phenomenon": str(record.get("phenomenon") or ""),
        "caption_sha256": hashlib.sha256(
            str(record.get("caption") or "").encode("utf-8")
        ).hexdigest(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    for key in ("source_dataset", "dataset_revision", "source_lineage_verified", "image_lineage",
                "upstream_image_sha256", "image_group_id"):
        if key in record:
            result[key] = record[key]
    if "explanation" in record:
        result["reference_explanation_sha256"] = hashlib.sha256(record["explanation"].encode("utf8")).hexdigest()
    return result


def build_selection_manifest(records, strategy="stratified", seed=42, dataset_split=None, cohort=None):
    """Build one full ordered manifest whose prefixes define every run size."""
    ordered = select_records(records, len(records), strategy=strategy, seed=seed)
    entries = [record_manifest_entry(record) for record in ordered]
    body = {
        "schema_version": "1.0",
        "selection_algorithm": {"stratified": "nested-stratified-v1", "random": "seeded-shuffle-v1", "prefix": "ordered-prefix-v1"}[strategy],
        "strategy": strategy,
        "seed": int(seed),
        "records": entries,
    }
    if dataset_split is not None:
        body.update(schema_version="2.0", dataset_split=dataset_split, cohort=cohort)
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body


def select_from_manifest(records, manifest, count):
    """Validate a manifest against local data and return its requested prefix."""
    unsigned = dict(manifest or {})
    declared_hash = str(unsigned.pop("manifest_sha256", ""))
    observed_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not declared_hash or declared_hash != observed_hash:
        raise ValueError("Selection manifest checksum is invalid.")
    by_id = {str(record.get("id")): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate source IDs")
    entries = list((manifest or {}).get("records", []) or [])
    identifiers = [str(entry.get("id") or "") for entry in entries]
    if len(identifiers) != len(set(identifiers)) or any(not key for key in identifiers):
        raise ValueError("Selection manifest IDs must be nonempty and unique")
    if not entries:
        raise ValueError("Selection manifest contains no records.")
    if type(count) is not int or not 0 < count <= len(entries):
        raise ValueError("Requested count exceeds the cohort or is not positive; no silent truncation")
    selected = []
    for entry in entries[:count]:
        sample_id = str(entry.get("id") or "")
        if sample_id not in by_id:
            raise ValueError(f"Selection manifest sample is missing: {sample_id}")
        observed = record_manifest_entry(by_id[sample_id])
        for key in ("label", "phenomenon", "caption_sha256", "image_sha256", *(
            key for key in ("source_dataset", "dataset_revision", "source_lineage_verified", "image_lineage",
                            "upstream_image_sha256", "image_group_id", "reference_explanation_sha256") if key in entry)):
            if str(entry.get(key)) != str(observed.get(key)):
                raise ValueError(
                    f"Selection manifest mismatch for {sample_id}: {key}"
                )
        selected.append(by_id[sample_id])
    return selected
