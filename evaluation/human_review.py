"""Prepare blinded human-review materials; never synthesize human ratings."""
import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

RATINGS = ("roles_preserved", "negation_preserved", "quantities_preserved", "scope_preserved",
           "visual_premise_correct", "relation_correct", "citation_support", "explanation_faithful")


def blind_record(row, caption, image_file):
    trace = row.get("trace") or {}
    language = trace.get("language_output") or {}
    artifact = trace.get("final_artifact") or {}
    return {"audit_id": hashlib.sha256(str(row["id"]).encode()).hexdigest()[:20],
            "source_caption": caption, "image_file": image_file,
            "claim": {key: language.get(key) for key in (
                "caption_proposition", "claim_subject", "claim_predicate", "claim_object",
                "negation", "quantities", "claim_modifiers", "comparison_direction",
                "time_or_panel_scope", "expected_visual_state", "opposite_visual_state")},
            "evidence": [{key: item.get(key) for key in ("id", "text", "relation", "question_id", "derived_from_ids")}
                         for item in artifact.get("cited_evidence", [])],
            "supporting_observations": [{key: item.get(key) for key in
                ("id", "text", "relation", "region", "panel", "derived_from_ids")}
                for item in artifact.get("supporting_ancestors", [])],
            "proposed_answer_for_faithfulness_review": artifact.get("final_label"),
            "explanation": artifact.get("delivered_explanation"),
            "ratings": {key: None for key in RATINGS}, "rater_id": None, "notes": "",
            "gold_hidden": True, "method_identity_hidden": True,
            "answer_visible_only_because_explanation_faithfulness_requires_it": True}


def make_packets(records, raw_by_id, output, seed=42):
    target = Path(output)
    if target.exists():
        raise FileExistsError("Choose a new human-review output directory")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate audit cases")
    packet, images, private = [], {}, []
    from PIL import Image
    for row in records:
        raw = raw_by_id[row["id"]]
        image_hash = hashlib.sha256(raw["image_bytes"]).hexdigest()
        caption_hash = hashlib.sha256(raw["caption"].encode()).hexdigest()
        if row.get("image_sha256") != image_hash or row.get("caption_sha256") != caption_hash:
            raise ValueError("Annotation input does not match the run")
        audit_id = hashlib.sha256(str(row["id"]).encode()).hexdigest()[:20]
        with Image.open(BytesIO(raw["image_bytes"])) as img:
            extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(img.format, "img")
        image_file = audit_id + "." + extension
        images[image_file] = raw["image_bytes"]  # exact source bytes, no re-encoding
        packet.append(blind_record(row, raw["caption"], image_file))
        private.append({"audit_id": audit_id, "sample_id": row["id"], "image_sha256": image_hash,
                        "caption_sha256": caption_hash})
    target.mkdir(parents=True)
    for filename, content in images.items():
        (target / filename).write_bytes(content)
    for slot in (1, 2):
        ordered = deepcopy(packet)
        random.Random(seed + slot).shuffle(ordered)
        (target / f"rater_{slot}.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    (target / "private_mapping_DO_NOT_GIVE_RATERS.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
    (target / "INSTRUCTIONS.txt").write_text(
        "Rate each dimension independently as true, false, or null when genuinely unassessable. "
        "Enter your own rater_id and explain uncertainty in notes. Do not discuss ratings before "
        "independent submission. The shown answer is the system output, not the gold label. "
        "Judge faithfulness to the cited evidence, not eloquence. Do not consult the private mapping.",
        encoding="utf-8")


def agreement(left, right):
    a, b = [{row["audit_id"]: row for row in rows} for rows in (left, right)]
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Human audit IDs must be unique and paired")
    if any(not a[key].get("rater_id") or not b[key].get("rater_id")
           or a[key]["rater_id"] == b[key]["rater_id"] for key in a):
        raise ValueError("Two distinct identified human raters are required")
    result = {}
    for dimension in RATINGS:
        pairs = [(a[key]["ratings"][dimension], b[key]["ratings"][dimension]) for key in a]
        if any(value is not None and type(value) is not bool for pair in pairs for value in pair):
            raise ValueError("Ratings must be true, false, or null")
        known = [(x, y) for x, y in pairs if x is not None and y is not None]
        observed = sum(x == y for x, y in known) / len(known) if known else None
        p = sum(x for x, _ in known) / len(known) if known else 0
        q = sum(y for _, y in known) / len(known) if known else 0
        expected = p * q + (1 - p) * (1 - q)
        result[dimension] = {"paired_known": len(known), "unassessable_pairs": len(pairs) - len(known),
                             "agreement": observed,
                             "cohen_kappa": (observed - expected) / (1 - expected) if known and expected < 1 else None}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from dataset.loaders import load_split
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    make_packets(rows, {row["id"]: row for row in load_split(args.split)}, args.output)
