"""Download V-FLUTE and build the three processed splits used by FigDebate."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from numbers import Integral
import os
from pathlib import Path
import pickle

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image


from dataset.protocol import (DATASET_ID, DATASET_REVISION, PROTOCOL_VERSION,
                              included, validate_subset, image_identity)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "data" / "processed"
DEV_MANIFEST = (
    PROJECT_ROOT / "dataset" / "data" / "splits" / "vflute_train_dev50.json"
)
EXPECTED_SIZES = {
    "vflute_train": 3992,
    "vflute_train_dev50": 50,
    "vflute_val": 573,
    "vflute_test": 569,
}


def normalize_label(value, label_feature=None) -> str:
    candidate = value
    if isinstance(value, Integral) and label_feature is not None:
        int2str = getattr(label_feature, "int2str", None)
        if callable(int2str):
            candidate = int2str(value)

    normalized = str(candidate).strip().upper()
    aliases = {
        "ENTAILMENT": "ENTAILS",
        "ENTAILS": "ENTAILS",
        "CONTRADICTION": "CONTRADICTS",
        "CONTRADICTS": "CONTRADICTS",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported V-FLUTE label: {value!r}")
    return aliases[normalized]


def encode_image(image_value) -> bytes:
    if isinstance(image_value, dict) and image_value.get("bytes"):
        return bytes(image_value["bytes"])
    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, dict) and image_value.get("path"):
        image = Image.open(image_value["path"])
    else:
        raise TypeError(f"Unsupported image value: {type(image_value).__name__}")

    output = BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def build_record(row, sample_id: str, label_feature=None) -> dict:
    caption = row.get("claim", row.get("caption"))
    if caption is None:
        raise KeyError("V-FLUTE row has neither 'claim' nor 'caption'.")
    image_bytes = encode_image(row["image"])
    return {
        "id": sample_id,
        "source_dataset": row.get("source_dataset", "UNKNOWN_UPSTREAM"),
        "upstream_dataset_id": DATASET_ID,
        "upstream_split": sample_id.rsplit("_", 1)[0].removeprefix("vflute_"),
        "native_label": row["label"],
        "upstream_row_index": int(sample_id.rsplit("_", 1)[1]),
        "dataset_revision": DATASET_REVISION,
        "dataset_protocol": PROTOCOL_VERSION,
        **image_identity(image_bytes),
        "source": "vflute",
        "phenomenon": str(row["phenomenon"]).strip().lower(),
        "image_bytes": image_bytes,
        "caption": str(caption),
        "label": normalize_label(row["label"], label_feature),
        "explanation": str(row.get("explanation", "")),
    }


def atomic_pickle(records: list[dict], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)


def validate_records(name: str, records: list[dict]) -> None:
    validate_subset(records)
    expected = EXPECTED_SIZES[name]
    if len(records) != expected:
        raise ValueError(f"{name} has {len(records)} rows; expected {expected}.")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate sample IDs.")
    invalid = sorted({record["label"] for record in records} - {
        "ENTAILS", "CONTRADICTS"
    })
    if invalid:
        raise ValueError(f"{name} contains invalid labels: {invalid}")


def validate_pickle(name: str, path: Path) -> None:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"{name} must contain a list of records.")
    validate_records(name, records)


def prepare(force: bool = False, output_dir=None) -> None:
    if force:
        raise ValueError("In-place replacement is disabled. Use a new --output-dir for a versioned rebuild.")
    destination = Path(output_dir).resolve() if output_dir else PROCESSED_DIR
    if output_dir and destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Versioned output directory must be new or empty")
    targets = {
        name: destination / f"{name}.pkl" for name in EXPECTED_SIZES
    }
    pending = set()
    for name, path in targets.items():
        if not path.exists():
            pending.add(name)
            continue
        try:
            validate_pickle(name, path)
        except Exception as error:
            raise ValueError(
                f"Existing {path.name} is invalid ({error}); refusing silent replacement. "
                "Inspect the data and use a new --output-dir for an intentional versioned rebuild."
            ) from error
    if not pending:
        print("V-FLUTE files pass structural checks only; source and semantic qualification are separate.")
        return

    from datasets import load_dataset, Image as DatasetImage

    print(f"Downloading {DATASET_ID} for: {', '.join(sorted(pending))}")

    if "vflute_train_dev50" in pending:
        train = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
        train = train.cast_column("image", DatasetImage(decode=False))
        label_feature = train.features.get("label")
        with DEV_MANIFEST.open("r", encoding="utf-8") as handle:
            selected = json.load(handle)
        records = []
        for expected in selected:
            sample_id = expected["id"]
            index = int(sample_id.rsplit("_", 1)[1])
            record = build_record(train[index], sample_id, label_feature)
            if record["caption"] != expected["caption"]:
                raise ValueError(f"Caption mismatch for locked sample {sample_id}.")
            if record["label"] != expected["label"]:
                raise ValueError(f"Label mismatch for locked sample {sample_id}.")
            records.append(record)
        validate_records("vflute_train_dev50", records)
        atomic_pickle(records, targets["vflute_train_dev50"])

    for output_name, source_split in (
        ("vflute_train", "train"),
        ("vflute_val", "validation"),
        ("vflute_test", "test"),
    ):
        if output_name not in pending:
            continue
        dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=source_split)
        dataset = dataset.cast_column("image", DatasetImage(decode=False))
        label_feature = dataset.features.get("label")
        records = [
            build_record(row, f"vflute_{source_split}_{index}", label_feature)
            for index, row in enumerate(dataset) if included(row)
        ]
        validate_records(output_name, records)
        atomic_pickle(records, targets[output_name])

    print(f"Prepared V-FLUTE data in {destination}; this does not establish semantic qualification.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Deprecated: in-place replacement is refused."
    )
    parser.add_argument("--output-dir", help="New versioned destination; never overwrites legacy splits")
    args = parser.parse_args()
    prepare(force=args.force, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
