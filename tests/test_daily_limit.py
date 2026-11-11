"""Daily Gemini cap. It sits in front of the 15-request limiter."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rag.engine import RAGConfig, RAGEngine
from rag.gemini import (
    MIN_START_INTERVAL_SECONDS,
    DailyLimitConfigError,
    DailyLimitError,
    GeminiClient,
    GeminiRequestError,
    configured_daily_limit,
    seconds_until_utc_midnight,
    shared_daily_limit,
    shared_gemini_limiter,
)

NOON = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
MODEL = "gemini-3.5-flash-lite"


def _sdk(text="Answer: yes"):
    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if isinstance(text, Exception):
                raise text
            return type("Response", (), {"text": text})()

    class SDK:
        def __init__(self):
            self.models = Models()

    return SDK()


def _client(sdk):
    return GeminiClient(api_key="local-test-value", sdk_client=sdk)


def _freeze(moment):
    shared_daily_limit().bind_clock(lambda: moment)


def test_missing_and_blank_limits_are_off():
    assert configured_daily_limit({}) is None
    assert configured_daily_limit({"RAG_DAILY_LIMIT": "   "}) is None
    assert configured_daily_limit({"RAG_DAILY_LIMIT": "  2  "}) == 2


@pytest.mark.parametrize("raw", ["0", "nope", "1.5", "-3"])
def test_invalid_limit_names_the_variable_and_not_the_value(raw):
    with pytest.raises(DailyLimitConfigError, match="^RAG_DAILY_LIMIT must be a positive integer$") as caught:
        configured_daily_limit({"RAG_DAILY_LIMIT": raw})
    assert raw not in str(caught.value)


def test_retry_after_is_the_wait_until_utc_midnight():
    assert seconds_until_utc_midnight(NOON) == 12 * 60 * 60
    almost = datetime(2026, 11, 11, 23, 59, 59, 200000, tzinfo=timezone.utc)
    assert seconds_until_utc_midnight(almost) == 1
    vietnam = timezone(timedelta(hours=7))
    six_am_vietnam = datetime(2026, 11, 12, 6, 0, tzinfo=vietnam)
    assert seconds_until_utc_midnight(six_am_vietnam) == 60 * 60


def test_cap_rejects_before_the_rate_limiter(monkeypatch, fake_gemini_clock):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "1")
    _freeze(NOON)
    sdk = _sdk()
    client = _client(sdk)

    first = client.generate("one", MODEL)

    assert first.rate_limit_wait_seconds == 0
    assert sdk.models.calls == 1
    granted = shared_gemini_limiter().granted_times()
    with pytest.raises(DailyLimitError) as caught:
        client.generate("two", MODEL)

    assert caught.value.retry_after_seconds == 12 * 60 * 60
    assert sdk.models.calls == 1
    assert shared_gemini_limiter().granted_times() == granted
    assert fake_gemini_clock.sleeps == []


def test_calls_under_the_cap_still_use_the_short_limiter(monkeypatch, fake_gemini_clock):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "5")
    sdk = _sdk()
    client = _client(sdk)

    first = client.generate("one", MODEL)
    second = client.generate("two", MODEL)

    assert first.rate_limit_wait_seconds == 0
    assert second.rate_limit_wait_seconds == pytest.approx(MIN_START_INTERVAL_SECONDS)
    assert sdk.models.calls == 2
    assert fake_gemini_clock.sleeps == pytest.approx([MIN_START_INTERVAL_SECONDS])


def test_a_failed_gemini_call_consumes_the_daily_cap(monkeypatch, fake_gemini_clock):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "1")
    failing = _sdk(TimeoutError("deadline exceeded"))
    with pytest.raises(GeminiRequestError):
        _client(failing).generate("one", MODEL)
    follower = _sdk()
    with pytest.raises(DailyLimitError):
        _client(follower).generate("two", MODEL)

    assert failing.models.calls == 1
    assert follower.models.calls == 0
    assert len(shared_gemini_limiter().granted_times()) == 1
    assert fake_gemini_clock.sleeps == []


def test_the_cap_resets_on_the_next_utc_day_and_the_limiter_still_spaces(monkeypatch, fake_gemini_clock):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "1")
    clock = {"now": datetime(2026, 11, 11, 23, 0, tzinfo=timezone.utc)}
    shared_daily_limit().bind_clock(lambda: clock["now"])
    sdk = _sdk()
    client = _client(sdk)

    client.generate("one", MODEL)
    with pytest.raises(DailyLimitError) as caught:
        client.generate("two", MODEL)
    assert caught.value.retry_after_seconds == 60 * 60
    assert fake_gemini_clock.sleeps == []

    clock["now"] = datetime(2026, 11, 12, 0, 0, tzinfo=timezone.utc)
    third = client.generate("three", MODEL)

    assert third.rate_limit_wait_seconds == pytest.approx(MIN_START_INTERVAL_SECONDS)
    assert sdk.models.calls == 2
    assert len(shared_gemini_limiter().granted_times()) == 2


def test_invalid_limit_does_not_call_gemini(monkeypatch):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "0")
    sdk = _sdk()
    with pytest.raises(DailyLimitConfigError):
        _client(sdk).generate("one", MODEL)
    assert sdk.models.calls == 0
    assert shared_gemini_limiter().granted_times() == []


class _Vector:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)

    def embed(self, texts):
        return self.vector


def _write_artifact(directory):
    import faiss

    rows = [
        {"chunk_id": 0, "pubid": 111, "text": "alpha passage"},
        {"chunk_id": 1, "pubid": 222, "text": "beta passage"},
        {"chunk_id": 2, "pubid": 333, "text": "gamma passage"},
    ]
    matrix = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(matrix)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "index.faiss"))
    with (directory / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "model": {"id": "BAAI/bge-small-en-v1.5", "requested_revision": "model-sha"},
        "pooling": "cls",
        "normalization": "l2",
        "metric": "inner_product",
        "index": "IndexFlatIP",
        "dtype": "float32",
        "dimension": 3,
        "num_chunks": 3,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_out_of_scope_does_not_consume_the_daily_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "1")
    sdk = _sdk()
    client = _client(sdk)
    _write_artifact(tmp_path)
    engine = RAGEngine(
        RAGConfig(artifact_dir=tmp_path, device="cpu", top_k=1, min_score=0.68),
        client=client,
    )
    engine.load_resources(embedder=_Vector([0.50, 0.0, 0.0]))

    result = engine.answer("Is the piano in tune?")

    assert result["label"] == "out_of_scope"
    assert sdk.models.calls == 0
    client.generate("still allowed", MODEL)
    assert sdk.models.calls == 1
    with pytest.raises(DailyLimitError):
        client.generate("over the cap", MODEL)
    assert sdk.models.calls == 1
