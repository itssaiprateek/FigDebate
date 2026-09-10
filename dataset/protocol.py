"""Locked three-phenomenon protocol and input identity (no label inference)."""
from collections import Counter
from io import BytesIO
import hashlib
import json
import random

DATASET_ID = "ColumbiaNLP/V-FLUTE"
DATASET_REVISION = "d920d848e21b6aba96e03704b134c546f89b74f4"
PHENOMENA = frozenset({"humor", "metaphor", "sarcasm"})
PROTOCOL_VERSION = "vflute-three-phenomena-1"


def included(row):
    return str(row.get("phenomenon", "")).strip().casefold() in PHENOMENA


def image_identity(image_bytes):
    from PIL import Image
    with Image.open(BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
        pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(gray.getdata())
        bits = [values[y * 9 + x] > values[y * 9 + x + 1] for y in range(8) for x in range(8)]
        dhash = sum(int(bit) << index for index, bit in enumerate(bits))
    return {"image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_group_id": pixels, "image_dhash": f"{dhash:016x}"}


def validate_subset(records):
    ids = [str(row["id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs are not allowed.")
    if any(not included(row) for row in records):
        raise ValueError("Only humor, metaphor and sarcasm are in the locked protocol.")
    if any(row.get("label") not in {"ENTAILS", "CONTRADICTS"} for row in records):
        raise ValueError("Unexpected relation label; no automatic relabeling is permitted.")


def manifest(records, split):
    validate_subset(records)
    entries = []
    for row in records:
        identity = image_identity(row["image_bytes"])
        entries.append({"id": row["id"], "phenomenon": row["phenomenon"],
            "source_dataset": row.get("source_dataset", "UNKNOWN_LEGACY"),
            "caption_sha256": hashlib.sha256(row["caption"].encode()).hexdigest(),
            "label": row["label"], **identity})
    return {"protocol": PROTOCOL_VERSION, "dataset_id": DATASET_ID,
            "rebuild_revision": DATASET_REVISION, "split": split,
            "rows": len(entries), "phenomena": dict(Counter(row["phenomenon"] for row in records)),
            "label_counts": dict(Counter(row["label"] for row in records)), "samples": entries,
            "legacy_revision_verified": all(row.get("dataset_revision") == DATASET_REVISION for row in records)}


def grouped_partition(entries, calibration_fraction=0.2, seed=42):
    """Internal partition by image identity, without selecting on outcomes."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one.")
    groups = sorted({row["image_group_id"] for row in entries})
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required.")
    random.Random(seed).shuffle(groups)
    size = max(1, min(len(groups) - 1, round(len(groups) * calibration_fraction)))
    calibration = set(groups[:size])
    return {"seed": seed, "selection_uses_labels": False,
            "calibration_ids": [row["id"] for row in entries if row["image_group_id"] in calibration],
            "development_ids": [row["id"] for row in entries if row["image_group_id"] not in calibration]}


def cross_split_overlaps(manifests):
    result = []
    for i, left in enumerate(manifests):
        for right in manifests[i + 1:]:
            a = {row["image_group_id"] for row in left["samples"]}
            b = {row["image_group_id"] for row in right["samples"]}
            result.append({"left": left["split"], "right": right["split"],
                           "shared_pixel_groups": sorted(a & b)})
    return result
