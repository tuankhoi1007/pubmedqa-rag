import threading
import time
from datetime import datetime, timezone

import pytest

from rag.gemini import shared_daily_limit, shared_gemini_limiter


class FakeClock:
    """Monotonic clock whose sleep only moves the clock."""

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep seconds must be non-negative")
        with self._lock:
            self.sleeps.append(seconds)
            self._now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


@pytest.fixture(autouse=True)
def fake_gemini_clock():
    clock = FakeClock()
    limiter = shared_gemini_limiter()
    limiter.bind_clock(clock.monotonic, clock.sleep)
    limiter.reset()
    yield clock
    limiter.bind_clock(time.monotonic, time.sleep)
    limiter.reset()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def reset_daily_limit(monkeypatch):
    monkeypatch.delenv("RAG_DAILY_LIMIT", raising=False)
    limiter = shared_daily_limit()
    limiter.bind_clock(_utc_now)
    limiter.reset()
    yield
    limiter.bind_clock(_utc_now)
    limiter.reset()
