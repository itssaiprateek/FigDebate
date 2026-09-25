"""Distinguish host suspension from active inference on Windows."""
import ctypes
import os
import time


def awake_seconds():
    if os.name != 'nt':
        return None
    try:
        value = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
            return value.value / 10_000_000
    except (AttributeError, OSError):
        pass
    return None


class HostInterrupted(TimeoutError):
    pass


class ExecutionClock:
    def __init__(self):
        self.started = time.perf_counter()
        self.awake_started = awake_seconds()

    def sample(self):
        elapsed = time.perf_counter() - self.started
        current = awake_seconds()
        active = None if current is None or self.awake_started is None else max(0., current - self.awake_started)
        suspended = None if active is None else max(0., elapsed - active)
        return {'wall_seconds': elapsed, 'awake_seconds': active, 'suspended_seconds': suspended,
                'host_interrupted': suspended is not None and suspended > 2.0,
                'suspension_detection_available': active is not None}
