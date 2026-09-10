"""Per-run actual generation requests; unknown failed-attempt tokens stay unknown."""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import time

_STATE = ContextVar("figdebate_generation_accounting", default=None)


def begin_accounting():
    _STATE.set({"events": [], "sample_id": None, "stage": None})


def set_scope(sample_id, stage, attempt=0):
    state = _STATE.get()
    if state is not None:
        state.update(sample_id=str(sample_id), stage=str(stage), stage_attempt=int(attempt))


def record_generation(model_family):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            started = time.perf_counter()
            self._last_generation_diagnostics = {}
            result, error, diagnostics = None, None, {}
            try:
                result = function(self, *args, **kwargs)
                return result
            except Exception as caught:
                error = {"type": type(caught).__name__, "message": str(caught)}
                raise
            finally:
                if isinstance(result, tuple) and len(result) == 4 and isinstance(result[-1], dict):
                    diagnostics = result[-1]
                else:
                    diagnostics = dict(getattr(self, "_last_generation_diagnostics", {}) or {})
                state = _STATE.get()
                if state is not None:
                    state["events"].append({
                        "event_id": len(state["events"]) + 1, "sample_id": state["sample_id"],
                        "stage": state["stage"], "stage_attempt": state.get("stage_attempt", 0),
                        "model_family": model_family, "wall_seconds": time.perf_counter() - started,
                        "execution_status": "FAILED" if error else "SUCCEEDED", "error": error,
                        "input_tokens": diagnostics.get("input_tokens"),
                        "output_tokens": diagnostics.get("generated_tokens"),
                        "failed_attempts": deepcopy(diagnostics.get("failed_attempts", [])),
                        "peak_allocated_gb": diagnostics.get("peak_allocated_gb"),
                    })
        return wrapped
    return decorate


def sample_accounting(sample_id):
    state = _STATE.get() or {"events": []}
    events = [deepcopy(row) for row in state["events"] if row["sample_id"] == str(sample_id)]
    complete = bool(events) and all(row["input_tokens"] is not None and row["output_tokens"] is not None
                                   and not row["failed_attempts"] for row in events)
    return {"scope": "executed_generation_requests_only_not_cached_work_or_scoring_forward_passes",
            "generation_requests": len(events),
            "input_tokens_known": sum(row["input_tokens"] or 0 for row in events),
            "output_tokens_known": sum(row["output_tokens"] or 0 for row in events),
            "generation_tokens_complete": complete, "compute_match_established": False,
            "events": events}
