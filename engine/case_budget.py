"""Cumulative time bound for a case's judge work, excluding other model stages."""
from contextlib import contextmanager
import time


class CaseBudgetExceeded(TimeoutError):
    pass


def remaining_seconds(runtime):
    deadline = getattr(runtime, "_review_budget_deadline", None)
    return None if deadline is None else max(0.0, deadline - time.perf_counter())


def require_time(runtime):
    remaining = remaining_seconds(runtime)
    if remaining is not None and remaining <= 0:
        raise CaseBudgetExceeded("judge case budget exhausted; no further generation permitted")
    return remaining


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
