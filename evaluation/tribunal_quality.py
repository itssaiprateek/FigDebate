"""Judge proposal quality and accepted-change quality, evaluated separately."""
import math


def summarize_tribunal(records, wall_seconds=None):
    counts = {k: 0 for k in ("samples", "initial_correct", "final_correct", "helpful_changes", "harmful_changes",
        "valid_binary_reviews", "semantic_abstentions", "failed_reviews", "correct_change_proposals",
        "harmful_change_proposals", "wrong_confirmations", "correct_confirmations", "rejected_correct_proposals", "reviews_not_requested")}
    times = []
    verification_failures = {}
    failed_verification_cases = 0
    invalid_verification_cases = 0
    for r in records:
        gold, initial, final = r.get("ground_truth"), r.get("initial_prediction"), r.get("prediction")
        if gold not in {"ENTAILS", "CONTRADICTS"}:
            raise ValueError("Quality evaluation requires a binary reference for every record")
        counts["samples"] += 1
        counts["initial_correct"] += initial == gold
        counts["final_correct"] += final == gold
        counts["helpful_changes"] += initial != gold and final == gold
        counts["harmful_changes"] += initial == gold and final != gold
        requested = r.get("judge_requested", True)
        counts["reviews_not_requested"] += not requested
        valid = requested and bool(r.get("judge_format_valid"))
        verdict = r.get("judge_verdict")
        binary = valid and verdict in {"ENTAILS", "CONTRADICTS"}
        counts["valid_binary_reviews"] += binary
        counts["semantic_abstentions"] += valid and verdict == "ABSTAIN"
        counts["failed_reviews"] += requested and not valid
        if binary:
            counts["correct_change_proposals"] += verdict != initial and verdict == gold
            counts["harmful_change_proposals"] += verdict != initial and verdict != gold and initial == gold
            counts["wrong_confirmations"] += verdict == initial and initial != gold
            counts["correct_confirmations"] += verdict == initial and initial == gold
            counts["rejected_correct_proposals"] += verdict != initial and verdict == gold and final != gold
        seconds = r.get("runtime_seconds")
        if seconds is not None and math.isfinite(float(seconds)):
            times.append(float(seconds))
        judge = r.get("trace", {}).get("judge", {})
        case_execution_failed = case_schema_failed = False
        pending = list(judge.get("tribunal_reviews", []))
        while pending:
            review = pending.pop()
            pending.extend(review.get("_verification_repair_history", []))
            proof = review.get("_independent_verification") or {}
            proof_calls = list(proof.get("calls", [])) + list(proof.get("obligations", {}).values())
            if review.get("_argument_repair"):
                proof_calls.append(review["_argument_repair"])
            case_execution_failed |= any(c.get("_execution_status") == "FAILED" for c in proof_calls)
            case_schema_failed |= any(c.get("_execution_status") == "SUCCEEDED" and c.get("_format_valid") is False for c in proof_calls)
            stage = proof.get("stopped_after")
            if stage:
                verification_failures[stage] = verification_failures.get(stage, 0) + 1
        failed_verification_cases += case_execution_failed
        invalid_verification_cases += case_schema_failed
    def ratio(n, d):
        return n / d if d else None
    def percentile(p):
        if not times:
            return None
        ordered = sorted(times)
        pos = (len(ordered) - 1) * p
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    proposals = counts["correct_change_proposals"] + counts["harmful_change_proposals"]
    changes = counts["helpful_changes"] + counts["harmful_changes"]
    tribunal_seconds = sum(t.get("inclusive_seconds") or 0.0 for r in records for t in r.get("tribunal_timing", []))
    return dict(counts, proposed_changes=proposals, accepted_changes=changes,
        proposal_precision=ratio(counts["correct_change_proposals"], proposals),
        initial_error_discovery=ratio(counts["correct_change_proposals"], counts["samples"] - counts["initial_correct"]),
        harmful_proposal_rate=ratio(counts["harmful_change_proposals"], counts["initial_correct"]),
        accepted_change_precision=ratio(counts["helpful_changes"], changes),
        net_correct_gain=counts["helpful_changes"] - counts["harmful_changes"],
        sample_median_seconds=percentile(.5), sample_p95_seconds=percentile(.95), wall_seconds=wall_seconds,
        tribunal_seconds=tribunal_seconds, tribunal_timing_is_subset_of_wall=True,
        verification_short_circuits=verification_failures,
        cases_with_failed_verification_execution=failed_verification_cases,
        cases_with_invalid_verification_output=invalid_verification_cases,
        interpretation="Diagnostic counts; no claim of generalization or independent model errors.")
