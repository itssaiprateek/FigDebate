"""Read-only view of fresh image-group-disjoint internal data; no split rewrite."""
import argparse
import hashlib
import json
from pathlib import Path
from dataset.loaders import load_split
from dataset.protocol import manifest
from engine.sampling import build_selection_manifest


def build(audit_path, partition):
    raw_audit = Path(audit_path).read_bytes()
    audit = json.loads(raw_audit)
    assignment = audit["internal_train_partition"]
    if "excluded_previously_used_or_external_groups" not in assignment:
        raise ValueError("Audit predates the inspected-image exclusion repair")
    training = load_split("vflute_train")
    current = manifest(training, "vflute_train")
    original = next(item for item in audit["splits"] if item["split"] == "vflute_train")
    if current != original:
        raise ValueError("Dataset identities changed after the audit")
    chosen = set(assignment[partition + "_ids"])
    rows = [row for row in training if row["id"] in chosen]
    if len(rows) != len(chosen):
        raise ValueError("Partition IDs missing from existing dataset")
    excluded = set(assignment["excluded_previously_used_or_external_groups"])
    if any(row["image_group_id"] in excluded for row in rows):
        raise ValueError("Partition overlaps previously used or external groups")
    selection = build_selection_manifest(rows, strategy="random", seed=assignment["seed"])
    provenance = {"partition": partition, "audit_sha256": hashlib.sha256(raw_audit).hexdigest(),
                  "sample_manifest_sha256": selection["manifest_sha256"], "dataset_mutated": False,
                  "rows": len(rows), "groups": len({row["image_group_id"] for row in rows})}
    return selection, provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--partition", choices=("calibration", "development"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    sidecar = output.with_suffix(".provenance.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Partition artifacts are immutable; choose a new output")
    selection, provenance = build(args.audit, args.partition)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))
