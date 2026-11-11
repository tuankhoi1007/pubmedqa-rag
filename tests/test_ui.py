"""Gradio UI tests. They use a fake engine: no model, no key, and no Gemini call."""

import html

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.gemini import DailyLimitError, GeminiRequestError
from rag.ui import OUT_OF_SCOPE_MESSAGE, VALIDATION_MESSAGE, answer_question, build_ui

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient`:starlette.exceptions.StarletteDeprecationWarning"
)

QUESTION = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
RAW = "Answer: yes\nSources: 1\n"
SECRET = "super-secret-value"


class FakeEngine:
    def __init__(self, answer_error=None, payload=None):
        self.questions = []
        self.answer_error = answer_error
        self.payload = _payload() if payload is None else payload

    def load_resources(self):
        return None

    def answer(self, question):
        self.questions.append(question)
        if self.answer_error is not None:
            raise self.answer_error
        return self.payload


def _payload(label="yes", text="Mitochondria remodel the leaf. a < b & c"):
    return {
        "label": label,
        "raw_response": RAW,
        "evidence": [
            {
                "rank": 1,
                "chunk_id": 4,
                "pubid": 21645374,
                "text": text,
                "score": 0.797184,
            }
        ],
        "latency_seconds": 0.2,
        "rate_limit_wait_seconds": 0.0,
        "generation_seconds": 0.15,
        "sources": [1],
        "prompt_id": "exp10",
    }


def _ready_app(engine=None, max_pending=4):
    app = create_app(engine or FakeEngine(), max_pending=max_pending)
    app.state.ready = True
    return app


def test_build_ui_limits_the_ask_event_to_one_call():
    app = _ready_app()
    demo = build_ui(app)
    limits = [getattr(fn, "concurrency_limit", None) for fn in demo.fns.values()]
    assert 1 in limits
    assert demo._queue.default_concurrency_limit == 1


def test_handler_returns_label_sources_and_raw_response():
    engine = FakeEngine()
    app = _ready_app(engine)
    notice, label, latency, sources, raw = answer_question(app, f"  {QUESTION}  ")
    assert notice == ""
    assert "YES" in label
    assert "answer-yes" in label
    assert "Latency: 0.200 seconds" in latency
    assert "Rank 1" in sources
    assert "https://pubmed.ncbi.nlm.nih.gov/21645374/" in sources
    assert "0.797184" in sources
    assert "a &lt; b &amp; c" in sources
    assert "<script>" not in sources
    assert raw == RAW
    assert engine.questions == [QUESTION]


def test_out_of_scope_shows_the_gray_label_and_guidance():
    payload = _payload(label="out_of_scope")
    payload["sources"] = []
    payload["top1_score"] = 0.5
    engine = FakeEngine(payload=payload)
    app = _ready_app(engine)
    notice, label, latency, sources, raw = answer_question(app, QUESTION)
    assert notice == ""
    assert "OUT OF SCOPE" in label
    assert "answer-out-of-scope" in label
    assert html.escape(OUT_OF_SCOPE_MESSAGE) in label
    assert "Latency: 0.200 seconds" in latency
    assert "No cited sources." in sources
    assert raw == RAW


def test_parse_fail_is_shown_as_a_label():
    engine = FakeEngine(payload=_payload(label="parse_fail"))
    engine.payload["sources"] = []
    app = _ready_app(engine)
    _notice, label, _latency, sources, _raw = answer_question(app, QUESTION)
    assert "PARSE FAIL" in label
    assert "No cited sources." in sources


@pytest.mark.parametrize(
    ("error", "snippet"),
    [
        (ValueError(SECRET), "cannot be answered"),
        (GeminiRequestError("Gemini request failed", kind="rate_limit"), "too many requests"),
        (DailyLimitError(3600), "daily question limit"),
        (GeminiRequestError("Gemini request failed", kind="timeout"), "too long"),
        (GeminiRequestError("Gemini request failed", kind="other"), "answering service failed"),
        (RuntimeError(SECRET), "Something went wrong"),
        (TimeoutError(SECRET), "Something went wrong"),
    ],
)
def test_handler_shows_a_message_and_does_not_raise(error, snippet):
    engine = FakeEngine(answer_error=error)
    app = _ready_app(engine)
    notice, label, latency, sources, raw = answer_question(app, QUESTION)
    assert snippet.lower() in notice.lower()
    assert SECRET not in notice
    assert "Traceback" not in notice
    assert label == latency == sources == raw == ""


def test_not_ready_and_full_queue_do_not_call_the_engine():
    engine = FakeEngine()
    app = create_app(engine, max_pending=1)
    notice, *_rest = answer_question(app, QUESTION)
    assert "still starting" in notice
    assert engine.questions == []

    app.state.ready = True
    assert app.state.admission.try_enter() is True
    notice, *_rest = answer_question(app, QUESTION)
    assert "busy" in notice
    assert engine.questions == []
    app.state.admission.leave()


@pytest.mark.parametrize("text", ["", "   ", "x" * 1001])
def test_invalid_question_is_rejected_before_the_engine(text):
    engine = FakeEngine()
    app = _ready_app(engine)
    notice, label, latency, sources, raw = answer_question(app, text)
    assert VALIDATION_MESSAGE in notice
    assert label == latency == sources == raw == ""
    assert engine.questions == []


def test_ui_page_loads_and_root_redirects():
    engine = FakeEngine()
    with TestClient(create_app(engine, max_pending=1)) as client:
        root = client.get("/", follow_redirects=False)
        page = client.get("/ui", follow_redirects=False)
        slashed = client.get("/ui/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"].endswith("/ui")
    assert page.status_code == 200
    assert slashed.status_code == 200
    assert '<base href="/ui/" />' in page.text
    assert "PubMedQA" in page.text
    assert "yes/no" in page.text.lower() or "yes, no" in page.text.lower()
