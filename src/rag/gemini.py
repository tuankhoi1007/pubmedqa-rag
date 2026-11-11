"""Gemini adapter for one serving request.

The notebook batch loop retried 5 times, slept 20 seconds on HTTP 429, and
slept 5 seconds on every other error. This module does not do that.

Serving policy in use:
- timeout 30 seconds, the notebook ``HttpOptions(timeout=30000)``
- one attempt, and that attempt acquires the shared rate limit immediately
  before the SDK call
- ``generate_content(model=..., contents=prompt)`` with no temperature and no
  other generation config
- a failed request raises ``GeminiRequestError`` and still consumes its slot
- no sleep after the response

The process shares one limiter for this quota: at most 15 starts in any 60
seconds, and at least 4.1 seconds between starts. Fourteen gaps of 4.1 seconds
place the fifteenth start 57.4 seconds after the first, inside the window.
This only paces one process. It does not see another notebook or app on the
same key, does not enforce token quotas, and does not prevent every HTTP 429.
The model and API key are never swapped to avoid a limit.

``RAG_DAILY_LIMIT`` is a separate gate in front of this limiter. It counts
Gemini calls in this process and, when the count reaches the limit, rejects
the call until 00:00 UTC without taking a limiter slot. A blank or missing
value leaves the gate open. ``out_of_scope`` never reaches Gemini, so it does
not count. The gate resets when the UTC date changes and when the process
restarts.

Proposed limit if a retry is added later, still not the notebook loop:
- at most 2 attempts
- each attempt acquires its own rate-limit slot
- retry only a timeout or HTTP 429
- wait at most 1 second before that second attempt, in addition to the limiter
- then raise ``GeminiRequestError``
- do not turn either failure into ``maybe`` or ``parse_fail``
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from typing import Literal

SERVING_TIMEOUT_MS = 30_000
SERVING_MAX_ATTEMPTS = 1
DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"
QUOTA_MAX_REQUESTS = 15
QUOTA_WINDOW_SECONDS = 60.0
MIN_START_INTERVAL_SECONDS = 4.1


ErrorKind = Literal["timeout", "rate_limit", "other"]
_TIMEOUT_TYPES: tuple[type[BaseException], ...] | None = None


class GeminiRequestError(RuntimeError):
    """The Gemini request failed. This is not a parsed label.

    ``kind`` is ``timeout``, ``rate_limit``, or ``other``. The message stays
    generic. The original SDK exception remains on ``__cause__``.
    """

    def __init__(self, message: str, *, kind: ErrorKind) -> None:
        if kind not in ("timeout", "rate_limit", "other"):
            raise ValueError("kind must be timeout, rate_limit, or other")
        super().__init__(message)
        self.kind: ErrorKind = kind


def _timeout_types() -> tuple[type[BaseException], ...]:
    """Timeout classes that can be imported. A missing SDK module is skipped."""
    global _TIMEOUT_TYPES
    if _TIMEOUT_TYPES is not None:
        return _TIMEOUT_TYPES
    found: list[type[BaseException]] = [TimeoutError]
    for module_name, attr in (
        ("httpx", "TimeoutException"),
        ("httpx2", "TimeoutException"),
        ("google.genai._gaos.lib.compat_errors", "APITimeoutError"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        candidate = getattr(module, attr, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found.append(candidate)
    _TIMEOUT_TYPES = tuple(found)
    return _TIMEOUT_TYPES


def _structured_status(exc: BaseException) -> int | None:
    """HTTP status on the exception, if the SDK stored one as an int.

    The installed ``google.genai.errors.APIError`` stores that number on
    ``code``. ``status`` is a string such as ``RESOURCE_EXHAUSTED`` and is
    not a status code. ``status_code`` is checked for other client shapes.
    """
    for name in ("code", "status_code"):
        if not hasattr(exc, name):
            continue
        value = getattr(exc, name)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return value
    return None


def request_kind(exc: BaseException) -> ErrorKind:
    """Classify the original SDK exception. Timeout wins over a status code."""
    if isinstance(exc, _timeout_types()):
        return "timeout"
    status = _structured_status(exc)
    if status is not None:
        return "rate_limit" if status == 429 else "other"
    text = str(exc)
    if re.search(r"\b429\b", text) or "RESOURCE_EXHAUSTED" in text:
        return "rate_limit"
    return "other"


@dataclass(frozen=True)
class GenerationResult:
    """One Gemini attempt after it has left the rate limiter."""
 
    text: str
    rate_limit_wait_seconds: float
    generation_seconds: float


class GeminiRateLimiter:
    """Pace start times. Sleep happens outside the scheduling lock."""

    def __init__(
        self,
        *,
        min_interval: float = MIN_START_INTERVAL_SECONDS,
        max_requests: int = QUOTA_MAX_REQUESTS,
        window_seconds: float = QUOTA_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval < 0 or window_seconds <= 0:
            raise ValueError("rate limit intervals must be non-negative")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        self._min_interval = float(min_interval)
        self._max_requests = max_requests
        self._window_seconds = float(window_seconds)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()

    def bind_clock(
        self,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        with self._lock:
            self._clock = clock
            self._sleep = sleep

    def reset(self) -> None:
        with self._lock:
            self._starts.clear()

    def clock(self) -> float:
        return self._clock()

    def granted_times(self) -> list[float]:
        with self._lock:
            return list(self._starts)

    def scheduling_lock_is_held(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    def acquire(self) -> float:
        """Consume the next start slot. Return how long this caller waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                delay = self._earliest(now) - now
                if delay <= 1e-12:
                    self._starts.append(now)
                    return waited
            self._sleep(delay)
            waited += delay

    def _earliest(self, now: float) -> float:
        while self._starts and now - self._starts[0] >= self._window_seconds:
            self._starts.popleft()
        earliest = now
        if self._starts:
            earliest = max(earliest, self._starts[-1] + self._min_interval)
        if len(self._starts) >= self._max_requests:
            earliest = max(earliest, self._starts[0] + self._window_seconds)
        return earliest


_SHARED_LIMITER = GeminiRateLimiter()


def shared_gemini_limiter() -> GeminiRateLimiter:
    """The one limiter for this process. Adapters must not build another."""
    return _SHARED_LIMITER


class DailyLimitError(Exception):
    """The process daily cap rejected this Gemini call.

    ``retry_after_seconds`` is the whole seconds until 00:00 UTC. This is not
    a Gemini HTTP 429 and it does not consume a rate-limiter slot.
    """

    def __init__(self, retry_after_seconds: int) -> None:
        if (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, int)
            or retry_after_seconds < 1
        ):
            raise ValueError("retry_after_seconds must be a positive integer")
        self.retry_after_seconds = retry_after_seconds
        super().__init__("daily limit reached")


class DailyLimitConfigError(RuntimeError):
    """``RAG_DAILY_LIMIT`` is set but is not a positive integer."""


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def seconds_until_utc_midnight(now: datetime) -> int:
    """Whole seconds from ``now`` until the next 00:00 UTC. At least 1."""
    current = _as_utc(now)
    next_midnight = datetime.combine(
        current.date() + timedelta(days=1),
        datetime_time.min,
        tzinfo=timezone.utc,
    )
    remaining = (next_midnight - current).total_seconds()
    return max(1, math.ceil(remaining))


def configured_daily_limit(env: Mapping[str, str] | None = None) -> int | None:
    """Return the process cap, or ``None`` when the gate is off.

    Missing and blank values are off. Any other value must be a positive
    integer. The raw value is never included in the error.
    """
    source = os.environ if env is None else env
    raw = source.get("RAG_DAILY_LIMIT")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise DailyLimitConfigError("RAG_DAILY_LIMIT must be a positive integer")
    text = raw.strip()
    if not text:
        return None
    if not text.isdigit() or int(text) < 1:
        raise DailyLimitConfigError("RAG_DAILY_LIMIT must be a positive integer")
    return int(text)


class DailyCallLimit:
    """Count Gemini calls for the UTC day. One instance per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._day = None
        self._now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def bind_clock(self, now: Callable[[], datetime]) -> None:
        with self._lock:
            self._now = now

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._day = None

    def reserve(self, limit: int | None) -> None:
        """Take one call, or raise :class:`DailyLimitError` before any slot.

        ``limit is None`` leaves the counter unchanged. The check and the
        increment share one lock. Callers must not hold the rate-limiter lock.
        """
        if limit is None:
            return
        with self._lock:
            now = _as_utc(self._now())
            day = now.date()
            if self._day != day:
                self._day = day
                self._count = 0
            if self._count >= limit:
                raise DailyLimitError(seconds_until_utc_midnight(now))
            self._count += 1


_SHARED_DAILY = DailyCallLimit()


def shared_daily_limit() -> DailyCallLimit:
    """The one daily gate for this process."""
    return _SHARED_DAILY


def require_api_key(env: Mapping[str, str]) -> str:
    """Read ``GEMINI_API_KEY`` from the supplied mapping. Do not log the value."""
    value = env.get("GEMINI_API_KEY", "")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("GEMINI_API_KEY is not set")
    return value


class GeminiClient:
    """Call the Google GenAI SDK once. Pass ``sdk_client`` in tests."""

    def __init__(
        self,
        api_key: str,
        timeout_ms: int = SERVING_TIMEOUT_MS,
        sdk_client=None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is not set")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise ValueError("timeout_ms must be a positive integer")
        self._api_key = api_key
        self._timeout_ms = timeout_ms
        self._sdk_client = sdk_client

    def generate(self, prompt: str, model: str) -> GenerationResult:
        """One request. No retry and no generation config.

        The daily gate runs first. A rejection does not take a limiter slot.
        The shared limiter is acquired immediately before this attempt. A
        later retry loop must acquire again inside each attempt. The SDK
        timeout is unchanged and the scheduling lock is not held while the
        SDK runs.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        shared_daily_limit().reserve(configured_daily_limit())
        limiter = shared_gemini_limiter()
        rate_limit_wait_seconds = limiter.acquire()
        started = limiter.clock()
        try:
            response = self._sdk().models.generate_content(model=model, contents=prompt)
        except GeminiRequestError:
            raise
        except Exception as exc:
            raise GeminiRequestError("Gemini request failed", kind=request_kind(exc)) from exc
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            text = ""
        return GenerationResult(
            text=text,
            rate_limit_wait_seconds=rate_limit_wait_seconds,
            generation_seconds=limiter.clock() - started,
        )

    def _sdk(self):
        if self._sdk_client is not None:
            return self._sdk_client
        from google import genai
        from google.genai import types

        self._sdk_client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )
        return self._sdk_client


def gemini_client_from_environment() -> GeminiClient:
    """Build the real client. Called only when no client was injected."""
    return GeminiClient(api_key=require_api_key(os.environ), timeout_ms=SERVING_TIMEOUT_MS)
