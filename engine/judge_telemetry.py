"""Bounded diagnostic snapshots; never change hardware or inference settings."""
import ctypes
import os
import subprocess
import time
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_phase = "starting"


def mark_phase(phase):
    global _phase
    _phase = str(phase)


def measured_callback(callback, counters, name):
    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            counters[name] = counters.get(name, 0.0) + time.perf_counter() - started
    return measured


def runtime_snapshot():
    result = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_seconds": time.perf_counter(),
              "phase": _phase}
    if os.name == "nt":
        class Power(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
                       ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                       ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
        try:
            power = Power()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
                result["ac_line_status"] = {0: "battery", 1: "plugged_in"}.get(power.ac, "unknown")
        except (AttributeError, OSError):
            pass
    fields = ("pstate,clocks.current.sm,power.draw,temperature.gpu,utilization.gpu,memory.used,"
              "enforced.power.limit,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,"
              "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,"
              "clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons_counters.sw_power_cap,"
              "clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,"
              "clocks_event_reasons_counters.hw_power_brake_slowdown")
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if output.returncode == 0:
            result["gpus"] = [dict(zip(fields.split(","), line.split(", "))) for line in output.stdout.strip().splitlines()]
        else:
            result["gpu_telemetry"] = "unsupported_query:" + output.stderr.strip()[:180]
            basic = "pstate,clocks.current.sm,power.draw,temperature.gpu,utilization.gpu,memory.used"
            output = subprocess.run(["nvidia-smi", "--query-gpu=" + basic, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if output.returncode == 0:
                result["gpus"] = [dict(zip(basic.split(","), line.split(", "))) for line in output.stdout.strip().splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_telemetry"] = "unavailable"
    return result


@contextmanager
def monitor_run(directory, interval=3.0):
    """Read-only sampling across model handoffs; one joined worker per run."""
    stopped = threading.Event()
    path = Path(directory) / "hardware_timeline.jsonl"
    def sample():
        try:
            with path.open("a", encoding="utf-8") as stream:
                while not stopped.is_set():
                    row = runtime_snapshot()
                    if os.name == "nt":
                        try:
                            power = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                text=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                            row["active_power_scheme"] = power.stdout.strip()
                        except (OSError, subprocess.TimeoutExpired):
                            row["active_power_scheme"] = "unavailable"
                    try:
                        import psutil
                        row["cpu_percent"] = psutil.cpu_percent()
                        row["system_memory_available_bytes"] = psutil.virtual_memory().available
                        row["process_rss_bytes"] = psutil.Process().memory_info().rss
                    except (ImportError, OSError):
                        row["host_memory_telemetry"] = "unavailable"
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    stopped.wait(interval)
        except Exception as error:
            # Diagnostics must never cancel inference; failure remains visible.
            print(f"[telemetry-unavailable] {type(error).__name__}: {error}")
    worker = threading.Thread(target=sample, name="figdebate-hardware-telemetry", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=8)


def timeout_diagnosis(diagnostics):
    elapsed = diagnostics.get("elapsed_seconds", 0)
    first = diagnostics.get("first_token_seconds") or 0
    tokens = diagnostics.get("generated_tokens", 0)
    return {"effective_deadline_seconds": diagnostics.get("timeout_seconds"),
            "decode_tokens_per_second": max(0, tokens-1) / max(.001, elapsed-first),
            "cause": "UNDETERMINED_CHECK_HARDWARE_TIMELINE",
            "partial_output_usable": False}
