"""Cumulative time bound for a case's judge work, excluding other model stages."""
from contextlib import contextmanager
import time


class CaseBudgetExceeded(TimeoutError):
    pass


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
