"""Paired cluster resampling, treating image-caption siblings as dependent."""
from collections import defaultdict
import random


def verify_pair_identity(control, treatment, allow_legacy=False):
    if set(control) != set(treatment):
        raise ValueError("Paired runs must have identical sample IDs")
    complete = True
    for key, left in control.items():
        right = treatment[key]
        for field in ("image_sha256", "caption_sha256", "image_group_id"):
            a, b = left.get(field), right.get(field)
            if not a or not b:
                complete = False
                if not allow_legacy:
                    raise ValueError(f"Missing {field} for {key}; legacy comparison is diagnostic only")
            elif a != b:
                raise ValueError(f"Input identity differs: {field} for {key}")
        if left.get("phenomenon") != right.get("phenomenon"):
            raise ValueError(f"Phenomenon differs for {key}")
    return complete


def grouped_accuracy_difference(paired, seed=42, iterations=5000):
    if not paired or iterations < 100:
        raise ValueError("Nonempty pairs and at least 100 resamples required")
    groups = defaultdict(list)
    for row in paired:
        if not row.get("image_group_id"):
            raise ValueError("Missing image group")
        groups[row["image_group_id"]].append(
            int(row["treatment_correct"]) - int(row["control_correct"]))
    clusters = list(groups.values())
    total = sum(map(len, clusters))
    observed = sum(map(sum, clusters)) / total
    if len(clusters) < 2:
        return {"groups": len(clusters), "accuracy_delta": observed,
                "ci95": None, "permutation_p": None, "reason": "fewer_than_two_groups"}
    rng = random.Random(seed)
    samples, extreme = [], 0
    for _ in range(iterations):
        chosen = [rng.choice(clusters) for _ in clusters]
        samples.append(sum(map(sum, chosen)) / sum(map(len, chosen)))
        permuted = sum(sum(group) * rng.choice((-1, 1)) for group in clusters) / total
        extreme += abs(permuted) >= abs(observed) - 1e-12
    samples.sort()
    return {"groups": len(clusters), "rows": total, "accuracy_delta": observed,
            "ci95": [samples[int(.025 * iterations)], samples[int(.975 * iterations)]],
            "permutation_p": (extreme + 1) / (iterations + 1),
            "iterations": iterations, "seed": seed,
            "method": "paired_image_cluster_bootstrap_and_sign_flip",
            "assumption": "image groups are independent; near duplicates require separate audit"}
