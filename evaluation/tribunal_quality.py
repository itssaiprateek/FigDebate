"""Judge proposal quality and accepted-change quality, evaluated separately."""
import math


def summarize_tribunal(records, wall_seconds=None):
    tribunal_seconds = 0.0
    counts = {k: 0 for k in ("samples", "initial_correct", "final_correct", "helpful_changes", "harmful_changes",
        "valid_binary_reviews", "semantic_abstentions", "failed_reviews", "correct_change_proposals",
        "harmful_change_proposals", "wrong_confirmations", "correct_confirmations", "rejected_correct_proposals", "reviews_not_requested")}
    times = []
    verification_failures = {}
    failed_verification_cases = 0
    failed_feedback_cases = 0
    feedback_attempted_cases = 0
    invalid_verification_cases = 0
    initial_visual_diagnostics_cases = 0
    initial_visual_answers = 0
    failed_initial_visual_answers = 0
    cases_with_failed_initial_visual_answers = 0
    initial_visual_answer_errors = {}
    all_review_proposals = {"corrective": 0, "harmful": 0, "correct_confirmation": 0, "wrong_confirmation": 0}
    feedback_seconds = 0.0
    integrated_feedback_seconds = 0.0
    advisor_counts = {'attempted': 0, 'valid': 0, 'failed': 0, 'seconds': 0.0, 'issues': {}}
    followup_stops = {}
    terminal_outcomes, groups, directions = {}, {}, {}
    repair_transitions = {'attempted_cases': 0, 'new_correct_proposals': 0,
                         'accepted_helpful_after_repair': 0, 'accepted_harmful_after_repair': 0}
    tribunal_only_repairs = dict(repair_transitions)
    first_blocking_stages = {}
    for r in records:
        tribunal_seconds += sum(t.get('inclusive_seconds') or 0. for t in r.get('tribunal_timing', []))
        gold, initial, final = r.get("ground_truth"), r.get("initial_prediction"), r.get("prediction")
        if gold not in {"ENTAILS", "CONTRADICTS"}:
            raise ValueError("Quality evaluation requires a binary reference for every record")
        counts["samples"] += 1
        counts["initial_correct"] += initial == gold
        counts["final_correct"] += final == gold
        counts["helpful_changes"] += initial != gold and final == gold
        counts["harmful_changes"] += initial == gold and final != gold
        # Historical-run reporting only: the experimental advisor generator was removed.
        advisor = r.get('trace', {}).get('debate_details', {}).get('semantic_advisor')
        if advisor:
            advisor_counts['attempted'] += 1
            valid_advice = bool(advisor.get('_format_valid')) and advisor.get('_execution_status') == 'SUCCEEDED'
            advisor_counts['valid'] += valid_advice
            advisor_counts['failed'] += not valid_advice
            advisor_counts['seconds'] += advisor.get('_generation_seconds') or 0.0
            issue = advisor.get('issue', 'FAILED') if valid_advice else 'FAILED'
            advisor_counts['issues'][issue] = advisor_counts['issues'].get(issue, 0) + 1
        visual = r.get("trace", {}).get("visual_output", {}).get("_internal", {})
        if "atomic_answers" in visual:
            initial_visual_diagnostics_cases += 1
            answers = visual["atomic_answers"]
            initial_visual_answers += len(answers)
            failures = [answer for answer in answers if not answer.get("valid")]
            failed_initial_visual_answers += len(failures)
            cases_with_failed_initial_visual_answers += bool(failures)
            for answer in failures:
                error = answer.get("error") or answer.get("status") or "UNKNOWN"
                initial_visual_answer_errors[error] = initial_visual_answer_errors.get(error, 0) + 1
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
        terminal = (judge.get('tribunal_resolution') or {}).get('terminal_outcome', 'UNAVAILABLE_LEGACY_RECORD')
        terminal_outcomes[terminal] = terminal_outcomes.get(terminal, 0) + 1
        blocking = (judge.get('tribunal_resolution') or {}).get('first_blocking_stage')
        if blocking:
            first_blocking_stages[blocking] = first_blocking_stages.get(blocking, 0) + 1
        for table, key in ((groups, r.get('phenomenon', 'UNKNOWN')), (directions, str(initial) + ' -> ' + str(gold))):
            item = table.setdefault(key, {'samples': 0, 'initial_correct': 0, 'final_correct': 0,
                'helpful_changes': 0, 'harmful_changes': 0, 'correct_proposals': 0, 'harmful_proposals': 0})
            item['samples'] += 1
            item['initial_correct'] += initial == gold
            item['final_correct'] += final == gold
            item['helpful_changes'] += initial != gold and final == gold
            item['harmful_changes'] += initial == gold and final != gold
            item['correct_proposals'] += binary and verdict != initial and verdict == gold
            item['harmful_proposals'] += binary and verdict != initial and verdict != gold and initial == gold
        hearings = judge.get('tribunal_reviews') or []
        if any(h.get('_tribunal_repair', {}).get('attempted') and h.get('_tribunal_repair', {}).get('mode') == 'bounded' for h in hearings):
            tribunal_only_repairs['attempted_cases'] += 1
            before, after = hearings[0].get('provisional_verdict'), hearings[-1].get('provisional_verdict')
            tribunal_only_repairs['new_correct_proposals'] += before != gold and after == gold and initial != gold
            tribunal_only_repairs['accepted_helpful_after_repair'] += initial != gold and final == gold
            tribunal_only_repairs['accepted_harmful_after_repair'] += initial == gold and final != gold
        if any(h.get('_integrated_feedback', {}).get('attempted') for h in hearings):
            repair_transitions['attempted_cases'] += 1
            before = hearings[0].get('provisional_verdict')
            after = hearings[-1].get('provisional_verdict')
            repair_transitions['new_correct_proposals'] += before != gold and after == gold and initial != gold
            repair_transitions['accepted_helpful_after_repair'] += initial != gold and final == gold
            repair_transitions['accepted_harmful_after_repair'] += initial == gold and final != gold
        case_execution_failed = case_schema_failed = False
        feedback_failed = feedback_attempted = False
        pending = list(judge.get("tribunal_reviews", []))
        while pending:
            review = pending.pop()
            pending.extend(review.get("_verification_repair_history", []))
            pending.extend(review.get("_feedback_review_history", []))
            feedback = review.get('_integrated_feedback') or review.get("_precedent_feedback", {})
            if review.get('_integrated_feedback', {}).get('attempted'):
                integrated_feedback_seconds += review.get('_generation_seconds', 0) or 0
            feedback_seconds += feedback.get("guided_seconds", 0) or 0
            stopping = review.get("_follow_up_budget_status") or review.get("_follow_up_question_status")
            if stopping:
                followup_stops[stopping] = followup_stops.get(stopping, 0) + 1
            proposed = review.get("best_semantic_judgment", review.get("provisional_verdict"))
            if review.get("_format_valid") and proposed in {"ENTAILS", "CONTRADICTS"}:
                kind = ("corrective" if proposed == gold else "harmful") if proposed != initial else (
                    "correct_confirmation" if proposed == gold else "wrong_confirmation")
                all_review_proposals[kind] += 1
            feedback_attempted |= bool(feedback.get("attempted"))
            feedback_failed |= bool(feedback.get("attempted") and (feedback.get("execution_status") != "SUCCEEDED"
                                                                   or not feedback.get("format_valid")))
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
        failed_feedback_cases += feedback_failed
        feedback_attempted_cases += feedback_attempted
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
    return dict(counts, proposed_changes=proposals, accepted_changes=changes,
        terminal_outcomes=terminal_outcomes, by_phenomenon=groups, by_initial_to_reference= directions,
        integrated_repair_observed_transitions=repair_transitions,
        tribunal_only_repair_observed_transitions=tribunal_only_repairs,
        first_blocking_stages=first_blocking_stages,
        tribunal_only_repair_causal_benefit='not_established_without_matched_repair_disabled_ablation',
        integrated_repair_causal_benefit='not_established_without_matched_repair_disabled_ablation',
        proposal_precision=ratio(counts["correct_change_proposals"], proposals),
        initial_error_discovery=ratio(counts["correct_change_proposals"], counts["samples"] - counts["initial_correct"]),
        harmful_proposal_rate=ratio(counts["harmful_change_proposals"], counts["initial_correct"]),
        accepted_change_precision=ratio(counts["helpful_changes"], changes),
        initial_error_correction_rate=ratio(counts["helpful_changes"], counts["samples"] - counts["initial_correct"]),
        harmful_flip_rate=ratio(counts["harmful_changes"], counts["initial_correct"]),
        net_accuracy_gain_percentage_points=(100 * (counts["helpful_changes"] - counts["harmful_changes"]) / counts["samples"]
                                             if counts["samples"] else None),
        net_correct_gain=counts["helpful_changes"] - counts["harmful_changes"],
        sample_median_seconds=percentile(.5), sample_p95_seconds=percentile(.95), wall_seconds=wall_seconds,
        tribunal_seconds=tribunal_seconds, tribunal_timing_is_subset_of_wall=True,
        verification_short_circuits=verification_failures,
        cases_with_failed_verification_execution=failed_verification_cases,
        cases_with_feedback_attempt=feedback_attempted_cases,
        cases_with_failed_feedback_review=failed_feedback_cases,
        all_hearing_and_guided_proposals=all_review_proposals,
        guided_feedback_seconds=feedback_seconds,
        integrated_feedback_review_seconds=integrated_feedback_seconds,
        semantic_advisor=advisor_counts,
        semantic_advisor_accuracy_contribution='not_established_requires_matched_ablation',
        followup_stop_reasons=followup_stops,
        cases_with_invalid_verification_output=invalid_verification_cases,
        cases_with_initial_visual_diagnostics=initial_visual_diagnostics_cases,
        initial_visual_answers=initial_visual_answers,
        failed_initial_visual_answers=failed_initial_visual_answers,
        cases_with_failed_initial_visual_answers=cases_with_failed_initial_visual_answers,
        initial_visual_answer_errors=initial_visual_answer_errors,
        interpretation="Diagnostic counts; no claim of generalization or independent model errors.")
