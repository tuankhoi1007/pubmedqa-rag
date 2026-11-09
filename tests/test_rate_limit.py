import threading
import time

import pytest

from rag.gemini import (
    MIN_START_INTERVAL_SECONDS,
    QUOTA_MAX_REQUESTS,
    QUOTA_WINDOW_SECONDS,
    SERVING_TIMEOUT_MS,
    GeminiClient,
    GeminiRateLimiter,
    GeminiRequestError,
    shared_gemini_limiter,
)


def _sdk(text="Answer: yes", clock=None, events=None):
    class Models:
        def __init__(self):
            self.calls = 0
            self.kwargs = None

        def generate_content(self, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            if events is not None:
                events.append(("sdk",))
            if isinstance(text, Exception):
                raise text
            if clock is not None:
                clock.advance(0.2)
            return type("Response", (), {"text": text})()

    class SDK:
        def __init__(self):
            self.models = Models()

    return SDK()


def test_first_call_does_not_wait(fake_gemini_clock):
    client = GeminiClient(api_key="local-test-value", sdk_client=_sdk())
    started = time.perf_counter()

    result = client.generate("prompt", "gemini-3.5-flash-lite")

    assert result.rate_limit_wait_seconds == 0
    assert fake_gemini_clock.sleeps == []
    assert time.perf_counter() - started < 1


def test_consecutive_calls_are_spaced_by_at_least_4_1_seconds(fake_gemini_clock):
    sdk = _sdk()
    client = GeminiClient(api_key="local-test-value", sdk_client=sdk)
    started = time.perf_counter()

    first = client.generate("one", "gemini-3.5-flash-lite")
    second = client.generate("two", "gemini-3.5-flash-lite")

    assert first.rate_limit_wait_seconds == 0
    assert second.rate_limit_wait_seconds == pytest.approx(MIN_START_INTERVAL_SECONDS)
    assert fake_gemini_clock.sleeps == pytest.approx([MIN_START_INTERVAL_SECONDS])
    assert sdk.models.calls == 2
    assert time.perf_counter() - started < 1


def test_elapsed_interval_does_not_add_another_sleep(fake_gemini_clock):
    events = []

    def sleep(seconds):
        events.append(("sleep", seconds))
        fake_gemini_clock.sleep(seconds)

    shared_gemini_limiter().bind_clock(fake_gemini_clock.monotonic, sleep)
    client = GeminiClient(
        api_key="local-test-value",
        sdk_client=_sdk(clock=fake_gemini_clock, events=events),
    )

    client.generate("one", "gemini-3.5-flash-lite")
    fake_gemini_clock.advance(10)
    result = client.generate("two", "gemini-3.5-flash-lite")

    assert [name for name, *_rest in events] == ["sdk", "sdk"]
    assert result.rate_limit_wait_seconds == 0
    assert fake_gemini_clock.sleeps == []


def test_failed_call_still_consumes_a_slot(fake_gemini_clock):
    failing = GeminiClient(
        api_key="local-test-value",
        sdk_client=_sdk(TimeoutError("deadline exceeded")),
    )
    with pytest.raises(GeminiRequestError):
        failing.generate("prompt", "gemini-3.5-flash-lite")
    assert fake_gemini_clock.sleeps == []

    follower = GeminiClient(api_key="local-test-value", sdk_client=_sdk())
    result = follower.generate("prompt", "gemini-3.5-flash-lite")

    assert result.rate_limit_wait_seconds == pytest.approx(MIN_START_INTERVAL_SECONDS)
    assert fake_gemini_clock.sleeps == pytest.approx([MIN_START_INTERVAL_SECONDS])


def test_threads_cannot_take_the_same_slot(fake_gemini_clock):
    limiter = shared_gemini_limiter()
    started = time.perf_counter()
    threads = [
        threading.Thread(target=limiter.acquire)
        for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    granted = limiter.granted_times()
    assert granted[0] == pytest.approx(0)
    assert len(granted) == 5
    for earlier, later in zip(granted, granted[1:]):
        assert later - earlier >= MIN_START_INTERVAL_SECONDS - 1e-9
    assert time.perf_counter() - started < 1
    assert fake_gemini_clock.monotonic() == pytest.approx(granted[-1])


def test_adapters_share_one_limiter(fake_gemini_clock):
    assert shared_gemini_limiter() is shared_gemini_limiter()
    first = GeminiClient(api_key="local-test-value", sdk_client=_sdk())
    second = GeminiClient(api_key="local-test-value", sdk_client=_sdk())

    assert first.generate("one", "gemini-3.5-flash-lite").rate_limit_wait_seconds == 0
    assert second.generate("two", "gemini-3.5-flash-lite").rate_limit_wait_seconds == pytest.approx(
        MIN_START_INTERVAL_SECONDS
    )
    assert len(shared_gemini_limiter().granted_times()) == 2


def test_scheduling_lock_is_released_before_the_sdk_call(fake_gemini_clock):
    limiter = shared_gemini_limiter()

    class Models:
        def generate_content(self, **kwargs):
            assert limiter.scheduling_lock_is_held() is False
            return type("Response", (), {"text": "Answer: yes"})()

    class SDK:
        def __init__(self):
            self.models = Models()

    GeminiClient(api_key="local-test-value", sdk_client=SDK()).generate(
        "prompt",
        "gemini-3.5-flash-lite",
    )
    assert SERVING_TIMEOUT_MS == 30_000


def test_sliding_window_uses_its_own_clock():
    clock = _Clock()
    limiter = GeminiRateLimiter(
        min_interval=0,
        max_requests=2,
        window_seconds=10,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    assert limiter.acquire() == 0
    clock.advance(1)
    assert limiter.acquire() == 0
    assert clock.sleeps == []
    assert limiter.acquire() == pytest.approx(9)
    assert clock.sleeps == pytest.approx([9])
    assert QUOTA_MAX_REQUESTS == 15
    assert QUOTA_WINDOW_SECONDS == 60


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds
