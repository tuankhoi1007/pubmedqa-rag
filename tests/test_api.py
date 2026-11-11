"""HTTP API tests. They use a fake engine: no model, no key, and no Gemini call."""

import logging
import threading

import pytest
from fastapi.testclient import TestClient

from rag.api import DAILY_LIMIT_DETAIL, LOOPBACK_HOST, create_app, main
from rag.engine import RAGConfig, RAGEngine
from rag.gemini import DailyLimitConfigError, DailyLimitError, GeminiRequestError

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient`:starlette.exceptions.StarletteDeprecationWarning"
)

SECRET = "super-secret-value"
QUESTION = "Do mitochondria remodel lace plant leaves?"
LONG_TEXT = "M" * 480
RAW = "RAW-NOT-FOR-LOG Answer: yes\nSources: 1\n"


class Closeable:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


class FakeConfig:
    generation_model = "gemini-3.5-flash-lite"
    top_k = 10
    device = "cuda"
    prompt_id = "exp10"


class FakeEngine:
    def __init__(self, load_error=None, answer_error=None, payload=None):
        self._client = Closeable()
        self.config = FakeConfig()
        self._index = type("Index", (), {"ntotal": 3358})()
        self._manifest = {"model": {"id": "BAAI/bge-small-en-v1.5"}}
        self.loads = 0
        self.questions = []
        self.load_error = load_error
        self.answer_error = answer_error
        self.payload = _payload() if payload is None else payload
        self._loaded = False
        self.block = False
        self.release = threading.Event()
        self.entered = threading.Event()
        self.saw_running_loop = []
        self.inside = 0
        self.max_inside = 0
        self._lock = threading.Lock()

    @property
    def loaded(self):
        return self._loaded

    def load_resources(self):
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error
        self._loaded = True

    def answer(self, question):
        import asyncio

        try:
            asyncio.get_running_loop()
            self.saw_running_loop.append(True)
        except RuntimeError:
            self.saw_running_loop.append(False)
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        self.questions.append(question)
        self.entered.set()
        try:
            if self.block:
                assert self.release.wait(5)
            if self.answer_error is not None:
                raise self.answer_error
            result = self.payload
            return result(question) if callable(result) else result
        finally:
            with self._lock:
                self.inside -= 1


def _payload(label="yes", sources=None):
    return {
        "label": label,
        "raw_response": RAW,
        "evidence": [
            {
                "rank": 1,
                "chunk_id": 4,
                "pubid": 21645374,
                "text": LONG_TEXT,
                "score": 0.797184,
            },
            {
                "rank": 2,
                "chunk_id": 9,
                "pubid": 18222909,
                "text": "other passage",
                "score": 0.702123,
            },
        ],
        "latency_seconds": 0.2,
        "rate_limit_wait_seconds": 0.05,
        "generation_seconds": 0.15,
        "sources": [1] if sources is None else sources,
        "prompt_id": "exp10",
    }


def _app(engine, max_pending=4):
    return create_app(engine, max_pending=max_pending)


def test_health_ok_does_not_call_answer_and_load_happens_once():
    engine = FakeEngine()
    with TestClient(_app(engine)) as client:
        assert engine.loads == 1
        health = client.get("/health")
        first = client.post("/ask", json={"question": f"  {QUESTION}  "})
        second = client.post("/ask", json={"question": "A second question?"})
        assert engine.loads == 1
        assert engine._client.closes == 0
    assert engine._client.closes == 1
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "index_loaded": True,
        "num_vectors": 3358,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "generation_model": "gemini-3.5-flash-lite",
        "top_k": 10,
        "device": "cuda",
    }
    assert first.status_code == 200
    body = first.json()
    assert set(body) == {
        "label",
        "sources",
        "prompt_id",
        "latency_seconds",
        "rate_limit_wait_seconds",
        "generation_seconds",
    }
    assert body["label"] == "yes"
    assert body["prompt_id"] == "exp10"
    assert body["latency_seconds"] == 0.2
    assert body["rate_limit_wait_seconds"] == 0.05
    assert body["sources"] == [
        {
            "rank": 1,
            "chunk_id": 4,
            "pubid": "21645374",
            "score": 0.797184,
            "text": "M" * 300,
        }
    ]
    assert "raw_response" not in body
    assert "evidence" not in body
    assert RAW not in first.text
    assert engine.questions == [QUESTION, "A second question?"]
    assert engine.saw_running_loop == [False, False]
    assert second.status_code == 200


def test_debug_returns_raw_response_and_full_evidence():
    engine = FakeEngine()
    with TestClient(_app(engine)) as client:
        response = client.post("/ask?debug=true", json={"question": QUESTION})
    body = response.json()
    assert response.status_code == 200
    assert body["raw_response"] == RAW
    assert body["evidence"][0]["text"] == LONG_TEXT
    assert body["sources"][0]["text"] == "M" * 300
    assert body["sources"][0]["pubid"] == "21645374"
    assert body["evidence"][0]["pubid"] == 21645374
    assert "prompt" not in body


def test_out_of_scope_is_a_label_with_top1_score_and_no_sources():
    payload = _payload(label="out_of_scope", sources=[])
    payload["top1_score"] = 0.5
    engine = FakeEngine(payload=payload)
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    body = response.json()
    assert response.status_code == 200
    assert body["label"] == "out_of_scope"
    assert body["sources"] == []
    assert body["top1_score"] == 0.5
    assert body["rate_limit_wait_seconds"] == 0.05
    assert "raw_response" not in body


def test_parse_fail_stays_a_label():
    engine = FakeEngine(payload=_payload(label="parse_fail", sources=[]))
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["label"] == "parse_fail"
    assert response.json()["sources"] == []


def test_load_failure_reports_not_ready_and_ask_returns_503():
    engine = FakeEngine(load_error=FileNotFoundError("missing artifact"))
    with TestClient(_app(engine)) as client:
        health = client.get("/health")
        ask = client.post("/ask", json={"question": QUESTION})
    assert engine.loads == 1
    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert health.json()["index_loaded"] is False
    assert health.json()["num_vectors"] is None
    assert ask.status_code == 503
    assert ask.json() == {"error": "not_ready", "detail": "engine is not ready"}
    assert engine.questions == []
    assert engine._client.closes == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": "   "},
        {"question": ""},
        {"question": 12},
        {"question": "x" * 1001},
        {"question": QUESTION, "model": "other"},
    ],
)
def test_rejects_invalid_questions_without_calling_the_engine(body):
    engine = FakeEngine()
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json=body)
    assert response.status_code == 422
    assert engine.questions == []
    assert "Traceback" not in response.text


def test_question_at_the_length_limit_is_accepted():
    engine = FakeEngine()
    question = "y" * 1000
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": f"  {question}  "})
    assert response.status_code == 200
    assert engine.questions == [question]


def test_value_error_from_answer_is_422_without_the_exception_text():
    engine = FakeEngine(answer_error=ValueError(SECRET))
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_question", "detail": "question is invalid"}
    assert SECRET not in response.text
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (GeminiRequestError("Gemini request failed", kind="rate_limit"), 429, "rate_limit"),
        (GeminiRequestError("Gemini request failed", kind="timeout"), 504, "timeout"),
        (GeminiRequestError("Gemini request failed", kind="other"), 502, "upstream"),
    ],
)
def test_gemini_kind_maps_to_http_status(error, status, code):
    error.__cause__ = RuntimeError(SECRET)
    engine = FakeEngine(answer_error=error)
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == status
    assert response.json()["error"] == code
    assert response.headers.get("retry-after") is None
    assert SECRET not in response.text
    assert "Traceback" not in response.text
    assert "maybe" not in response.text


@pytest.mark.parametrize("error", [TimeoutError(SECRET), RuntimeError(SECRET), TypeError(SECRET)])
def test_unclassified_exceptions_are_500_without_a_traceback(error, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    caplog.set_level(logging.ERROR, logger="rag.api")
    engine = FakeEngine(answer_error=error)
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert response.json() == {"error": "internal", "detail": "request failed"}
    assert SECRET not in response.text
    assert "Traceback" not in response.text
    assert SECRET not in caplog.text
    assert "Traceback" in caplog.text


def test_runtime_error_after_load_is_not_treated_as_not_ready():
    engine = FakeEngine(answer_error=RuntimeError("search returned invalid chunk id 9"))
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert response.json()["error"] == "internal"
    assert "chunk" not in response.text


def test_full_queue_returns_503_with_retry_after():
    engine = FakeEngine()
    engine.block = True
    app = _app(engine, max_pending=1)
    results = []
    errors = []

    def post(client):
        try:
            response = client.post("/ask", json={"question": QUESTION})
            results.append((response.status_code, response.json(), response.headers.get("retry-after")))
        except Exception as exc:  # pragma: no cover - surfaces client races
            errors.append(exc)

    try:
        with TestClient(app) as client:
            first = threading.Thread(target=post, args=(client,))
            first.start()
            assert engine.entered.wait(2)
            second = threading.Thread(target=post, args=(client,))
            second.start()
            second.join(2)
            assert any(status == 503 for status, _, _ in results)
            assert engine.release.is_set() is False
            engine.release.set()
            first.join(5)
    finally:
        engine.release.set()
    assert errors == []
    statuses = sorted(status for status, _, _ in results)
    assert statuses == [200, 503]
    assert engine.max_inside == 1
    busy = [item for item in results if item[0] == 503]
    assert busy[0][1] == {"error": "queue_full", "detail": "server is busy"}
    assert busy[0][2] == "5"


def test_request_log_records_label_and_timings_without_the_answer(caplog):
    caplog.set_level(logging.INFO, logger="rag.api")
    engine = FakeEngine()
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    assert "label=yes" in caplog.text
    assert "latency_seconds=0.2" in caplog.text
    assert "rate_limit_wait_seconds=0.05" in caplog.text
    assert "RAW-NOT-FOR-LOG" not in caplog.text
    assert LONG_TEXT not in caplog.text


def test_dotenv_loads_once_and_does_not_override_the_environment(monkeypatch):
    calls = []

    def record_load_dotenv(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("rag.api._dotenv_loaded", False)
    monkeypatch.setattr("rag.api.load_dotenv", record_load_dotenv)
    create_app(FakeEngine(), max_pending=1)
    create_app(FakeEngine(), max_pending=1)
    assert calls == [((), {"override": False})]


def test_daily_limit_returns_429_with_retry_after():
    engine = FakeEngine(answer_error=DailyLimitError(43_200))
    with TestClient(_app(engine)) as client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 429
    assert response.json() == {"error": "daily_limit", "detail": DAILY_LIMIT_DETAIL}
    assert response.headers["retry-after"] == "43200"
    assert "rate_limit" not in response.text
    assert SECRET not in response.text


def test_invalid_daily_limit_refuses_to_start(monkeypatch):
    monkeypatch.setenv("RAG_DAILY_LIMIT", "nope")
    with pytest.raises(DailyLimitConfigError, match="^RAG_DAILY_LIMIT must be a positive integer$"):
        with TestClient(_app(FakeEngine())):
            pass


def test_rag_device_cpu_is_used_without_loading_the_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    app = create_app()
    assert app.state.engine.config.device == "cpu"
    assert app.state.engine.loaded is False


def test_blank_rag_device_keeps_cuda(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DEVICE", "   ")
    app = create_app()
    assert app.state.engine.config.device == "cuda"


def test_factory_does_not_load_or_read_the_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_DEVICE", raising=False)
    calls = {"count": 0}

    def forbid():
        calls["count"] += 1
        raise AssertionError("Gemini client was built")

    monkeypatch.setattr("rag.api.gemini_client_from_environment", forbid)
    app = create_app()
    engine = app.state.engine
    assert isinstance(engine, RAGEngine)
    assert engine.loaded is False
    assert engine.config == RAGConfig(
        artifact_dir="artifacts/pubmedqa-bge-small-v1",
        device="cuda",
    )
    assert calls["count"] == 0


def test_main_binds_loopback_with_one_worker_and_without_reload(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    main()
    assert captured["args"] == ("rag.api:app",)
    assert captured["kwargs"]["host"] == LOOPBACK_HOST == "127.0.0.1"
    assert captured["kwargs"]["workers"] == 1
    assert captured["kwargs"]["reload"] is False
    assert "0.0.0.0" not in str(captured)
