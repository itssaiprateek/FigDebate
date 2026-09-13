"""Bounded diagnostic snapshots; never change hardware or inference settings."""
import ctypes
import os
import subprocess
import time


def measured_callback(callback, counters, name):
    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            counters[name] = counters.get(name, 0.0) + time.perf_counter() - started
    return measured


def runtime_snapshot():
    result = {}
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
    fields = "pstate,clocks.current.sm,power.draw,temperature.gpu,utilization.gpu,memory.used"
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if output.returncode == 0:
            result["gpus"] = [dict(zip(fields.split(","), line.split(", "))) for line in output.stdout.strip().splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_telemetry"] = "unavailable"
    return result
