"""Three-phenomenon development smoke selection; never a performance study."""
import argparse
import json
from pathlib import Path

from dataset.loaders import load_split
from engine.sampling import build_selection_manifest


def prepare(path):
    target = Path(path)
    if target.exists():
        raise FileExistsError("Qualification manifests are immutable")
    data = load_split("vflute_train_dev50")
    # Prespecified raw-order first case of each phenomenon, without consulting
    # labels, predictions, explanations, or earlier qualification outcomes.
    cases = [next(row for row in data if row["phenomenon"] == kind)
             for kind in ("humor", "metaphor", "sarcasm")]
    manifest = build_selection_manifest(cases, strategy="prefix", seed=42)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": str(target.resolve()), "ids": [row["id"] for row in cases],
            "purpose": "functional_development_qualification_not_accuracy_inference"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest), indent=2))
