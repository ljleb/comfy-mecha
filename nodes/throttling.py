import time
import threading
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple


class RunLastAtMostEvery:
    """
    Coalescing (run-the-last) throttle.

    Goal:
      - You may call it as often as you want.
      - It will execute the wrapped function at most once every `interval_seconds`.
      - If multiple calls arrive during the interval, only the most recent one is kept.
      - Calls never block.
      - The function runs on a background thread (threading.Timer).

    Example:
      throttled = RunLastAtMostEvery(fn, 0.5)
      throttled(1); throttled(2); throttled(3)
      # Eventually runs fn(3), not fn(1)/fn(2)
    """

    def __init__(self, func: Callable[..., Any], interval_seconds: float):
        self.func = func
        self.interval_seconds = float(interval_seconds)

        self._lock = threading.Lock()
        self._pending_call: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None
        self._scheduled_timer: Optional[threading.Timer] = None
        self._earliest_next_start = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._pending_call = (args, kwargs)
            if self._scheduled_timer is None:
                self._schedule_when_allowed_locked()

    def _schedule_when_allowed_locked(self) -> None:
        now = time.monotonic()
        seconds_until_allowed = max(0.0, self._earliest_next_start - now)

        timer = threading.Timer(seconds_until_allowed, self._run_pending_call)
        timer.daemon = True
        self._scheduled_timer = timer
        timer.start()

    def _run_pending_call(self) -> None:
        with self._lock:
            call = self._pending_call
            self._pending_call = None
            self._scheduled_timer = None

            if call is None:
                return

            args, kwargs = call

            start_time = time.monotonic()
            self._earliest_next_start = start_time + self.interval_seconds

        try:
            self.func(*args, **kwargs)
        finally:
            with self._lock:
                if self._pending_call is not None and self._scheduled_timer is None:
                    self._schedule_when_allowed_locked()


def run_last_at_most_every(interval_seconds: float):
    """Decorator form of RunLastAtMostEvery."""
    def decorator(func: Callable[..., Any]):
        throttled = RunLastAtMostEvery(func, interval_seconds)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            throttled(*args, **kwargs)

        return wrapper
    return decorator
