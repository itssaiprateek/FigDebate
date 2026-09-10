"""Export every selected case for blinded human review, without inventing ratings."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
from dataset.loaders import attach_provenance
from dataset.protocol import validate_subset


def build(processed_dir, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    cases, key, files = [], [], []
    for split in ("vflute_train", "vflute_val", "vflute_test"):
        path = Path(processed_dir) / (split + ".pkl")
        with path.open("rb") as handle:
            before = hashlib.file_digest(handle, "sha256").hexdigest()
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        validate_subset(rows)
        rows = attach_provenance(rows, split, path)
        if not all(row.get("source_lineage_verified") for row in rows):
            raise ValueError("Source verification must be completed before annotation export")
        for row in rows:
            image_sha = hashlib.sha256(row["image_bytes"]).hexdigest()
            image_path = images / (image_sha + ".jpg")
            if not image_path.exists():
                image_path.write_bytes(row["image_bytes"])
            case_id = hashlib.sha256(("blind-dataset-review-v1:" + row["id"]).encode()).hexdigest()[:20]
            cases.append({"case_id": case_id, "image": "images/" + image_path.name,
                "caption": row["caption"], "rater_id": None, "image_readable": None,
                "relation": None, "requires_unavailable_context": None,
                "ambiguous": None, "visual_evidence": None, "reason": None})
            key.append({"case_id": case_id, "id": row["id"], "split": split,
                "source_dataset": row["source_dataset"], "native_label": row["label"],
                "reference_explanation": row["explanation"], "phenomenon": row["phenomenon"],
                "image_sha256": image_sha})
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != before:
                raise ValueError("Source changed during export")
        files.append({"split": split, "file_sha256": before, "rows": len(rows)})
    random.Random(42).shuffle(cases)
    for name, rows in (("blinded_cases.jsonl", cases), ("ADMIN_ONLY_gold_key.jsonl", key)):
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    instructions = """# Blinded dataset review

Give each of two independent reviewers a separate copy of blinded_cases.jsonl
and images/. Do NOT give reviewers ADMIN_ONLY_gold_key.jsonl or model outputs.
Review all cases in order; ratings are deliberately blank. No ratings have been
supplied by the export tool. Dev50 is already included inside train, not repeated.

For relation use ENTAILS, CONTRADICTS, or INSUFFICIENT_OR_AMBIGUOUS. The third
option is a review flag, not a new model output label or automatic gold change.
Mark image_readable, requires_unavailable_context and ambiguous true or false;
record visible evidence and an explanation. Lack of visible support must not
automatically become contradiction. Interpret figurative meaning using the
image and caption, and explicitly identify any unsupported assumptions.

Lock the rubric and independently annotate before viewing the gold key. Report
agreement before adjudication, all disagreement categories and sample counts.
A third reviewer adjudicates disagreements, retaining both original judgments.
Do not use model correctness to choose exclusions or replacement labels.
Keep official-source scores separate from any later adjudicated benchmark.
Readability flags alone do not prove corruption (e.g. intentional black images).

This packet is for annotation, not evidence that annotation has happened.
"""
    (output / "REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf8")
    summary = {"cases": len(cases), "images": len(list(images.iterdir())), "files": files,
               "all_ratings_blank": True, "publication_qualified": False,
               "blinded_cases_sha256": hashlib.sha256((output / "blinded_cases.jsonl").read_bytes()).hexdigest()}
    (output / "packet_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.processed_dir, args.output_dir), indent=2))
