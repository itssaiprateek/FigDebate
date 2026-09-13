
import os
import pickle
import json
import hashlib
from pathlib import Path
from io import BytesIO
from PIL import Image
from dataset.protocol import validate_subset, image_identity, DATASET_REVISION

EXPECTED_SPLITS = [
    "vflute_train", "vflute_train_dev50", "vflute_val", "vflute_test",
]

def _package_root():
    here = os.path.abspath(os.path.dirname(__file__))
    configured = os.environ.get("FIGDEBATE_DATA_ROOT")
    if configured:
        return os.path.abspath(configured)
    if os.path.isdir(os.path.join(here, "data", "processed")):
        return here
    parent = os.path.abspath(os.path.join(here, os.pardir))
    if os.path.isdir(os.path.join(parent, "data", "processed")):
        return parent
    # Importing pure validation helpers must not require a downloaded dataset.
    # Actual loads retain their explicit missing-file and provenance checks.
    return here

BASE = _package_root()
PROCESSED_DIR = os.path.join(BASE, "data", "processed")


def attach_provenance(records, split_name, path):
    """Recover source metadata only when both the corpus and sidecar hashes match."""
    directory = Path(BASE) / "data" / "provenance" / "vflute_d920d848"
    summary_path = directory / "verification.json"
    if not summary_path.exists():
        return records
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    if not summary.get("all_source_lineage_verified") or summary.get("revision") != DATASET_REVISION:
        raise ValueError("Dataset source verification is incomplete or refers to another revision")
    expected = (summary["dev50_file_sha256"] if split_name == "vflute_train_dev50" else
                next(s["local_file_sha256"] for s in summary["splits"] if s["split"] == split_name))
    with open(path, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError("Dataset bytes changed since upstream verification; re-audit before running")
    raw = (directory / "provenance.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["provenance_sha256"]:
        raise ValueError("Dataset provenance sidecar checksum mismatch")
    by_id = {r["id"]: r for r in map(json.loads, raw.splitlines())}
    enriched = []
    for row in records:
        source = by_id[row["id"]]
        if not source["lineage_verified"]:
            raise ValueError("Unverified source row")
        enriched.append(dict(row, **{key: source[key] for key in (
            "source_dataset", "dataset_revision", "upstream_dataset_id", "upstream_split",
            "upstream_row_index", "native_label", "upstream_image_sha256", "image_lineage",
            "text_transformations")}, source_lineage_verified=True, semantic_gold_reviewed=False))
    return enriched

def _load(split_name):
    if split_name not in EXPECTED_SPLITS:
        raise ValueError(f"Unknown split: {split_name}")
    path = os.path.join(PROCESSED_DIR, f"{split_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        records = pickle.load(f)
    validate_subset(records)
    records = attach_provenance(records, split_name, path)
    # Add identity in memory only; never rewrite historical split files.
    return [dict(row, **image_identity(row["image_bytes"])) for row in records]

def decode_image(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def load_split(split_name): return _load(split_name)
