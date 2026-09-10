"""Gold reference sidecars live outside inference inputs and are hash-bound to selection."""
import hashlib
import json
from pathlib import Path


def text_hash(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def build_references(selected, manifest):
    return {"schema_version": "1.0", "selection_manifest_sha256": manifest["manifest_sha256"],
        "usage": "POST_PREDICTION_EVALUATION_ONLY",
        "records": [{"id": row["id"], "reference_explanation": row["explanation"],
                     "reference_explanation_sha256": text_hash(row["explanation"])} for row in selected]}


def load_references(parent, manifest, expected):
    path = Path(parent) / "evaluation_references.json"
    if not path.exists():
        if manifest.get("schema_version") == "2.0":
            raise ValueError("Missing hash-bound evaluation references for this run")
        return None
    data = json.loads(path.read_text(encoding="utf8"))
    if data["selection_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Reference bank belongs to a different selection")
    rows = {r["id"]: r for r in data["records"]}
    if len(rows) != len(data["records"]) or set(rows) != set(expected):
        raise ValueError("Reference IDs must match all intended cases exactly")
    for key, row in rows.items():
        observed = text_hash(row["reference_explanation"])
        if not row["reference_explanation"].strip() or observed != row["reference_explanation_sha256"]:
            raise ValueError("Empty or modified reference explanation")
        declared = expected[key].get("reference_explanation_sha256")
        if declared is not None and observed != declared:
            raise ValueError("Reference explanation differs from selected dataset")
    return rows
