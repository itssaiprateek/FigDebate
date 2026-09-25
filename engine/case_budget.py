"""Cumulative time bound for a case's judge work, excluding other model stages."""
from contextlib import contextmanager
import time


class CaseBudgetExceeded(TimeoutError):
    pass


def budget_estimates(runtime):
    """Bounded estimates from execution durations only, never labels or case IDs."""
    profile = getattr(runtime, "hardware_profile", None)
    reserve = float(getattr(profile, "judge_verification_reserve_seconds", 120))
    if getattr(profile, 'tribunal_protocol', '') == 'evidence-review-5.0':
        # Predeclared costs: other samples never change review eligibility.
        # Cost observations remain telemetry for a future frozen calibration.
        return {'proof': reserve, 'proposal': 45.0, 'followup': reserve + 45.0}
    if not getattr(profile, "judge_adaptive_budget", False):
        return {"proof": reserve, "proposal": 45.0, "followup": max(90.0, reserve + 45)}
    import statistics
    samples = getattr(runtime, "_review_cost_samples", {})
    def estimate(name, default, low, high):
        history = samples.get(name, [])[-16:]
        return min(high, max(low, statistics.median(history) * 1.35 if history else default))
    proof = estimate("proof", 75.0, 45.0, max(45.0, reserve))
    proposal = estimate("proposal", 40.0, 25.0, 90.0)
    return {"proof": proof, "proposal": proposal, "followup": proof + proposal + 10.0}


def observe_cost(runtime, name, seconds):
    if seconds and seconds > 0:
        history = getattr(runtime, "_review_cost_samples", {})
        history[name] = (history.get(name, []) + [float(seconds)])[-16:]
        runtime._review_cost_samples = history


def proposal_reserve(runtime):
    profile = getattr(runtime, "hardware_profile", None)
    reserve = budget_estimates(runtime)["proof"]
    if getattr(profile, 'tribunal_protocol', '') == 'evidence-review-5.0' and getattr(runtime, '_review_repair_reserve', None) is not None:
        return runtime._review_repair_reserve
    if getattr(profile, "judge_adaptive_budget", False) and getattr(profile, 'tribunal_protocol', '') != 'evidence-review-5.0':
        remaining = remaining_seconds(runtime)
        if remaining is not None:
            # Avoid stranding half a case budget while a proposal cannot finish.
            # This allocates time, never waives a verification obligation.
            reserve = min(reserve, max(0.0, remaining * .5))
    return reserve


def remaining_seconds(runtime):
    deadlines = [getattr(runtime, key, None) for key in ("_review_budget_deadline", "_review_phase_deadline")]
    deadlines = [d for d in deadlines if d is not None]
    deadline = min(deadlines) if deadlines else None
    return None if deadline is None else max(0.0, deadline - time.perf_counter())


def require_time(runtime):
    remaining = remaining_seconds(runtime)
    if remaining is not None and remaining <= 0:
        raise CaseBudgetExceeded("judge case budget exhausted; no further generation permitted")
    return remaining


def restore_spent_budget(runtime, key, review):
    """A resumed second hearing keeps the first hearing's consumed allowance."""
    budget = review.get('_case_budget') or {}
    if budget.get('remaining_seconds') is None or budget.get('total_seconds') is None:
        return
    used = max(0., budget['total_seconds'] - budget['remaining_seconds'])
    from engine.runtime_accounting import judge_case_times
    shared = judge_case_times()
    if not hasattr(runtime, '_review_case_times'):
        runtime._review_case_times = {}
    times = shared if shared is not None else runtime._review_case_times
    times[key] = max(times.get(key, 0.), used)
    runtime._review_case_times[key] = times[key]


@contextmanager
def reserve_for_verification(runtime, reserve_seconds):
    """All proposal attempts/retrieval share the remainder above a proof reserve."""
    previous = getattr(runtime, "_review_phase_deadline", None)
    remaining = remaining_seconds(runtime)
    if remaining is not None and reserve_seconds:
        runtime._review_phase_deadline = time.perf_counter() + max(0.0, remaining - reserve_seconds)
    try:
        yield
    finally:
        runtime._review_phase_deadline = previous


@contextmanager
def case_budget(runtime, key, seconds):
    if seconds is None or getattr(runtime, "_review_budget_deadline", None) is not None:
        yield
        return
    if not hasattr(runtime, "_review_case_times"):
        runtime._review_case_times = {}
    from engine.runtime_accounting import judge_case_times
    shared_times = judge_case_times()
    times = shared_times if shared_times is not None else runtime._review_case_times
    used = times.get(key, 0.0)
    started = time.perf_counter()
    runtime._review_budget_deadline = started + max(0.0, float(seconds) - used)
    try:
        yield
    finally:
        times[key] = used + time.perf_counter() - started
        runtime._review_case_times[key] = times[key]
        runtime._review_budget_deadline = None
