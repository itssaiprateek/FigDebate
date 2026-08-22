"""Download V-FLUTE and build the three processed splits used by FigDebate."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from numbers import Integral
import os
from pathlib import Path
import pickle

from PIL import Image


DATASET_ID = "ColumbiaNLP/V-FLUTE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "data" / "processed"
DEV_MANIFEST = (
    PROJECT_ROOT / "dataset" / "data" / "splits" / "vflute_train_dev50.json"
)
EXPECTED_SIZES = {
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
        "0": "ENTAILS",
        "1": "CONTRADICTS",
        "ENTAILMENT": "ENTAILS",
        "ENTAILS": "ENTAILS",
        "CONTRADICTION": "CONTRADICTS",
        "CONTRADICTS": "CONTRADICTS",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported V-FLUTE label: {value!r}")
    return aliases[normalized]


def encode_image(image_value) -> bytes:
    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, dict) and image_value.get("bytes"):
        image = Image.open(BytesIO(image_value["bytes"]))
    elif isinstance(image_value, dict) and image_value.get("path"):
        image = Image.open(image_value["path"])
    else:
        raise TypeError(f"Unsupported image value: {type(image_value).__name__}")

    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=85)
    return output.getvalue()


def build_record(row, sample_id: str, label_feature=None) -> dict:
    caption = row.get("claim", row.get("caption"))
    if caption is None:
        raise KeyError("V-FLUTE row has neither 'claim' nor 'caption'.")
    return {
        "id": sample_id,
        "source": "vflute",
        "phenomenon": str(row["phenomenon"]).strip().lower(),
        "image_bytes": encode_image(row["image"]),
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


def prepare(force: bool = False) -> None:
    targets = {
        name: PROCESSED_DIR / f"{name}.pkl" for name in EXPECTED_SIZES
    }
    pending = set()
    for name, path in targets.items():
        if force or not path.exists():
            pending.add(name)
            continue
        try:
            validate_pickle(name, path)
        except Exception as error:
            print(f"Existing {path.name} is invalid ({error}); rebuilding it.")
            pending.add(name)
    if not pending:
        print("V-FLUTE processed splits are present and valid.")
        return

    from datasets import load_dataset

    print(f"Downloading {DATASET_ID} for: {', '.join(sorted(pending))}")

    if "vflute_train_dev50" in pending:
        train = load_dataset(DATASET_ID, split="train")
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
        ("vflute_val", "validation"),
        ("vflute_test", "test"),
    ):
        if output_name not in pending:
            continue
        dataset = load_dataset(DATASET_ID, split=source_split)
        label_feature = dataset.features.get("label")
        records = [
            build_record(row, f"vflute_{source_split}_{index}", label_feature)
            for index, row in enumerate(dataset)
        ]
        validate_records(output_name, records)
        atomic_pickle(records, targets[output_name])

    print(f"Prepared V-FLUTE data in {PROCESSED_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild existing processed files."
    )
    args = parser.parse_args()
    prepare(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
