"""Pure helpers for auditable, position-balanced binary scoring."""

import math


def position_balanced_relation_scores(forward_scores, reversed_scores):
    """Cancel A/B position preference by scoring both option orders.

    In the forward prompt A means support and B means conflict. In the reversed
    prompt A means conflict and B means support. Averaging corresponding log
    scores removes a model preference for a particular option letter.
    """
    for scores in (forward_scores, reversed_scores):
        if not isinstance(scores, dict) or not {"A", "B"}.issubset(scores):
            raise ValueError("Both orientations require numeric A and B scores.")
        if not all(math.isfinite(float(scores[key])) for key in ("A", "B")):
            raise ValueError("Choice scores must be finite numbers.")

    support_log_score = (
        float(forward_scores["A"]) + float(reversed_scores["B"])
    ) / 2.0
    conflict_log_score = (
        float(forward_scores["B"]) + float(reversed_scores["A"])
    ) / 2.0

    peak = max(support_log_score, conflict_log_score)
    support_exp = math.exp(support_log_score - peak)
    conflict_exp = math.exp(conflict_log_score - peak)
    total = support_exp + conflict_exp
    return {
        "ENTAILS": support_exp / total,
        "CONTRADICTS": conflict_exp / total,
        "support_log_score": support_log_score,
        "conflict_log_score": conflict_log_score,
    }


def evidence_adjusted_confidence(raw_confidence, evidence_quality):
    """Shrink an uncalibrated selected-label probability toward 0.5."""
    raw = min(1.0, max(0.5, float(raw_confidence)))
    quality = min(1.0, max(0.0, float(evidence_quality)))
    return 0.5 + (raw - 0.5) * quality
