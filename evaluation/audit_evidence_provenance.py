"""Audit atomic evidence verification against saved development traces."""

import argparse
import csv
import json
import os

from engine.evidence_ledger import RELATION_FOR_LABEL, build_evidence_ledger, evidence_ids
from engine.evidence_verifier import AtomicEvidenceVerifier


def audit(records_path, output_dir):
    verifier = AtomicEvidenceVerifier()
    rows = []
    with open(records_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            trace = record.get("trace", {}) or {}
            visual = trace.get("visual_output", {}) or {}
            language = trace.get("language_output", {}) or {}
            comparison = trace.get("comparison", {}) or {}
            ledger = build_evidence_ledger(visual, language, comparison)
            ledger, summary = verifier.verify(ledger, language, comparison)
            # Audit NLI candidate directions even though runtime correctly
            # keeps them out of the decision-grade ledger.
            support_ids = [
                item["id"] for item in ledger
                if item.get("verification", {}).get("candidate_relation") == "SUPPORT"
            ]
            conflict_ids = [
                item["id"] for item in ledger
                if item.get("verification", {}).get("candidate_relation") == "CONFLICT"
            ]
            expected = RELATION_FOR_LABEL.get(record.get("ground_truth"))
            observed = {
                relation for relation, ids in (
                    ("SUPPORT", support_ids), ("CONFLICT", conflict_ids)
                ) if ids
            }
            if not observed:
                outcome = "abstained"
            elif len(observed) > 1:
                outcome = "mixed"
            elif expected in observed:
                outcome = "aligned"
            else:
                outcome = "opposing"
            rows.append({
                "sample": record.get("sample"),
                "id": record.get("id"),
                "phenomenon": record.get("phenomenon"),
                "ground_truth": record.get("ground_truth"),
                "expected_relation": expected,
                "verification_outcome": outcome,
                "support_ids": " | ".join(support_ids),
                "conflict_ids": " | ".join(conflict_ids),
                "candidate_count": summary.get("candidate_count", 0),
                "verified_count": summary.get("verified_count", 0),
                "claim": summary.get("claim", ""),
                "ledger_json": json.dumps(ledger, ensure_ascii=True),
            })

    counts = {
        name: sum(row["verification_outcome"] == name for row in rows)
        for name in ("aligned", "opposing", "mixed", "abstained")
    }
    directional = counts["aligned"] + counts["opposing"]
    metrics = {
        "samples": len(rows),
        **counts,
        "direct_coverage_rate": (
            (len(rows) - counts["abstained"]) / len(rows) if rows else 0.0
        ),
        "directional_precision_against_gold": (
            counts["aligned"] / directional if directional else None
        ),
        "runtime_promotion_enabled": False,
        "promotion_reason": (
            "Generic NLI candidates require independent grounded corroboration."
        ),
        "minimum_relation_probability": verifier.MIN_RELATION_PROBABILITY,
        "minimum_relation_margin": verifier.MIN_RELATION_MARGIN,
        "verifier_model": verifier.nli.MODEL_ID,
        "verifier_revision": verifier.nli.REVISION,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "evidence_verifier_audit.csv"),
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(
        os.path.join(output_dir, "evidence_verifier_audit.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Audit evidence verifier thresholds on saved development traces."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.records, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
