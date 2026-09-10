"""Frozen ablation recipes and blind human-audit packets; no invented ratings."""
import argparse
import hashlib
import json
from pathlib import Path
import random

CONDITIONS = {
    "no_debate": {"debate_mode": "disabled", "judge_mode": "disabled", "semantic_bridge_mode": "disabled"},
    "debate_without_tribunal": {"debate_mode": "enabled", "judge_mode": "disabled", "semantic_bridge_mode": "disabled"},
    "tribunal_shadow_bridge": {"debate_mode": "enabled", "judge_mode": "tribunal", "semantic_bridge_mode": "shadow"},
    "full_tribunal": {"debate_mode": "enabled", "judge_mode": "tribunal", "semantic_bridge_mode": "corroborated"},
    "judge_without_hearing": {"debate_mode": "disabled", "judge_mode": "tribunal", "semantic_bridge_mode": "corroborated", "candidate_mode": "disabled"},
    "tribunal_without_candidates": {"debate_mode": "enabled", "judge_mode": "tribunal", "semantic_bridge_mode": "corroborated", "candidate_mode": "disabled"},
    "tribunal_without_bridge": {"debate_mode": "enabled", "judge_mode": "tribunal", "semantic_bridge_mode": "disabled"},
    "independent_vote": {"debate_mode": "disabled", "judge_mode": "disabled", "semantic_bridge_mode": "disabled", "control_mode": "independent_vote"},
    "conventional_debate": {"debate_mode": "disabled", "judge_mode": "disabled", "semantic_bridge_mode": "disabled", "control_mode": "conventional_debate"},
}


def run_matrix(manifest_path, output_root, seeds=(42, 43, 44), dataset_split="vflute_train_dev50", num_samples=50):
    """Produce commands only; callers explicitly launch experiments."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("manifest_sha256"):
        raise ValueError("A locked selection manifest is required")
    return [{"condition": name, "seed": seed, "manifest_sha256": manifest["manifest_sha256"],
             "arguments": ["run_figdebate.py", "--dataset-split", dataset_split,
                 "--num-samples", str(num_samples), "--sample-manifest", str(Path(manifest_path).resolve()),
                 "--selection-strategy", manifest["strategy"],
                 "--selection-seed", str(manifest["seed"]), "--seed", str(seed), "--hardware-profile", "paper-8gb",
                 "--feedback-mode", "disabled", "--judge-scope", "all",
                 "--run-dir", str(Path(output_root) / f"{name}_seed{seed}")]
                 + [item for key, value in settings.items() for item in ("--" + key.replace("_", "-"), value)]}
            for seed in seeds for name, settings in CONDITIONS.items()]


def human_packet(records, seed=42):
    """Raters see raw inputs and claim/evidence fields, not gold or predictions."""
    packet = []
    for row in records:
        full = row.get("trace", {}) or {}
        language = full.get("language_output", {}) or {}
        packet.append({"audit_id": hashlib.sha256(str(row["id"]).encode()).hexdigest()[:16],
            "sample_id": row["id"], "source_caption": language.get("source_caption_raw", language.get("original_caption", "")), "image_sha256": row.get("image_sha256"),
            "caption_sha256": row.get("caption_sha256"),
            "claim_fields": {key: language.get(key) for key in (
                "claim_subject", "claim_predicate", "claim_object", "caption_proposition",
                "expected_visual_state", "opposite_visual_state", "intended_meaning")},
            "ratings": {key: None for key in ("roles_preserved", "negation_preserved",
                "quantities_preserved", "scope_preserved", "visual_premise_correct",
                "relation_correct", "citation_support", "explanation_faithful")},
            "rater_id": None, "notes": "", "gold_and_prediction_hidden": True})
    random.Random(seed).shuffle(packet)
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run_matrix(args.manifest, args.output_root), indent=2))
