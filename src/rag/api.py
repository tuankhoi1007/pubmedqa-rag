"""Local HTTP API in front of :class:`rag.engine.RAGEngine`.

Not medical advice. Retrieval, the exp10 prompt, Gemini generation, and the
shared rate limiter are unchanged.

Run one worker from the repository root::

    .\\.venv\\Scripts\\python.exe -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --workers 1

``GeminiRateLimiter`` is shared only inside one process. A second worker would
keep its own 15 requests per 60 seconds and the two together would pass the
quota. The command binds ``127.0.0.1`` and does not auto-reload.

``GET /health`` does not call Gemini and does not report quota. ``POST /ask``
is a sync endpoint so FastAPI runs ``answer()`` in its threadpool. At most
``RAG_MAX_PENDING`` calls (default 4) may be waiting or running; the next one
gets 503 and ``Retry-After`` immediately. ``GET /`` redirects to the Gradio UI
at ``/ui``. The UI calls the same ``ask_service`` and the same admission limit.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from rag.engine import RAGConfig, RAGEngine
from rag.gemini import GeminiRequestError, gemini_client_from_environment

DEFAULT_ARTIFACT_DIR = Path("artifacts/pubmedqa-bge-small-v1")
DEFAULT_DEVICE = "cuda"
LOOPBACK_HOST = "127.0.0.1"
MAX_QUESTION_CHARS = 1000
DEFAULT_MAX_PENDING = 4
SOURCE_TEXT_LIMIT = 300
RETRY_AFTER_SECONDS = "5"
logger = logging.getLogger("rag.api")
_dotenv_loaded = False

_QUERY_SECRET = re.compile(
    r"(?i)\b((?:api_key|x-goog-api-key|key)\s*=\s*)([^\s&\"']+)"
)


class _RedactSecrets(logging.Filter):
    """Drop the API key and key-like query values from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not isinstance(key, str):
            key = ""
        record.msg = _redact(record.getMessage(), key.strip())
        record.args = ()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = _redact(record.exc_text, key.strip())
            record.exc_info = None
        return True


def _redact(text: str, key: str) -> str:
    if key:
        text = text.replace(key, "[REDACTED]")
    return _QUERY_SECRET.sub(r"\1[REDACTED]", text)


def _load_local_dotenv() -> None:
    """Read a local ``.env`` once. Existing environment variables stay put."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    load_dotenv(override=False)
    _dotenv_loaded = True


def _install_log_filter() -> None:
    if any(isinstance(item, _RedactSecrets) for item in logger.filters):
        return
    logger.addFilter(_RedactSecrets())
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


class Admission:
    """Count of /ask calls that have been accepted and have not finished."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_pending must be a positive integer")
        self._limit = limit
        self._lock = threading.Lock()
        self._pending = 0

    def try_enter(self) -> bool:
        with self._lock:
            if self._pending >= self._limit:
                return False
            self._pending += 1
            return True

    def leave(self) -> None:
        with self._lock:
            if self._pending > 0:
                self._pending -= 1


class AskServiceError(Exception):
    """A failed ask call. ``detail`` is safe to show to a caller."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        *,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.retry_after = retry_after


class AskRequest(BaseModel):
    """One question. Generation settings stay on the engine."""

    model_config = ConfigDict(extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be a non-empty string")
        if len(stripped) > MAX_QUESTION_CHARS:
            raise ValueError("question exceeds the length limit")
        return stripped


class PassageModel(BaseModel):
    """One retrieved chunk. ``text`` may already be shortened."""

    model_config = ConfigDict(extra="ignore")

    rank: int
    chunk_id: int
    pubid: int | str
    score: float
    text: str

    @field_validator("pubid", mode="before")
    @classmethod
    def keep_pubid(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("pubid must be an int or a string")
        return value


class SourceModel(BaseModel):
    """A cited passage. ``pubid`` is text so JSON does not emit a number."""

    model_config = ConfigDict(extra="ignore")

    rank: int
    chunk_id: int
    pubid: str
    score: float
    text: str

    @field_validator("pubid", mode="before")
    @classmethod
    def pubid_as_text(cls, value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("pubid must be an int or a string")
        return str(value)


class AskResponse(BaseModel):
    """Public answer. Full evidence and the raw model text require debug=true."""

    model_config = ConfigDict(extra="ignore")

    label: Literal["yes", "no", "maybe", "parse_fail", "out_of_scope"]
    sources: list[SourceModel]
    prompt_id: str
    latency_seconds: float
    rate_limit_wait_seconds: float
    generation_seconds: float
    top1_score: float | None = None
    raw_response: str | None = None
    evidence: list[PassageModel] | None = None


class HealthResponse(BaseModel):
    """Local engine status. This request does not call Gemini."""

    status: Literal["ok", "not_ready"]
    index_loaded: bool
    num_vectors: int | None
    embedding_model: str | None
    generation_model: str | None
    top_k: int | None
    device: str | None


class _EnvironmentGemini:
    """Build the real Gemini client on the first generate call, not at import."""

    def __init__(self) -> None:
        self._client = None

    def generate(self, prompt: str, model: str):
        if self._client is None:
            self._client = gemini_client_from_environment()
        return self._client.generate(prompt, model)

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return
        await _invoke_close(client)
        await _invoke_close(getattr(client, "_sdk_client", None))


def create_app(
    engine: RAGEngine | object | None = None,
    *,
    config: RAGConfig | None = None,
    max_pending: int | None = None,
) -> FastAPI:
    """Build the app. Pass ``engine`` in tests so startup does not load a model.

    ``app.state.ready`` is the only "engine is loaded" signal used by ``/ask``.
    A later ``RuntimeError`` from ``answer()`` is an internal error, not 503.
    """
    _load_local_dotenv()
    _install_log_filter()
    limit = _pending_limit(max_pending)
    if engine is None:
        if config is None:
            config = RAGConfig(artifact_dir=DEFAULT_ARTIFACT_DIR, device=DEFAULT_DEVICE)
        if not isinstance(config, RAGConfig):
            raise TypeError("config must be a RAGConfig")
        engine = RAGEngine(config, client=_EnvironmentGemini())
    elif config is not None:
        raise ValueError("pass either an engine or a config, not both")

    app = FastAPI(
        title="PubMedQA RAG",
        summary="Research demo. Not medical advice.",
        debug=False,
        lifespan=_lifespan,
    )
    app.state.engine = engine
    app.state.ready = False
    app.state.admission = Admission(limit)

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Application status",
        description=(
            "Reports whether this process loaded the index. "
            "It does not call Gemini and does not check quota."
        ),
    )
    def health():
        body = _health(app.state.engine, app.state.ready)
        if body.status != "ok":
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.post(
        "/ask",
        response_model=AskResponse,
        response_model_exclude_none=True,
        summary="Answer one question",
    )
    def ask(body: AskRequest, debug: bool = False):
        try:
            return ask_service(app.state, body.question, debug)
        except AskServiceError as exc:
            return _error(
                exc.status_code,
                exc.code,
                exc.detail,
                retry_after=exc.retry_after,
            )

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse(url="/ui", status_code=307)

    from rag.ui import build_ui

    import gradio as gr

    blocks = build_ui(app)
    gr.mount_gradio_app(
        app,
        blocks,
        path="/ui",
        theme=blocks.pubmed_theme,
        css=blocks.pubmed_css,
        server_name=LOOPBACK_HOST,
        show_error=False,
        footer_links=[],
        ssr_mode=False,
        pwa=False,
    )
    # Starlette only matches the mount at /ui/. Serve the same page at /ui
    # with a base URL so relative Gradio assets still resolve.
    mounted_ui = _mounted_ui_app(app)

    @app.get("/ui", include_in_schema=False)
    async def ui_page(request: Request):
        status, body = await _read_mounted_page(mounted_ui, request)
        if status == 200:
            body = body.replace(b'<base href="./" />', b'<base href="/ui/" />', 1)
        media = "text/html; charset=utf-8" if status == 200 else "text/plain"
        return HTMLResponse(content=body, status_code=status, media_type=media)

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    engine = app.state.engine
    try:
        try:
            engine.load_resources()
        except Exception:
            app.state.ready = False
            logger.exception("RAG engine failed to start")
        else:
            app.state.ready = True
        yield
    finally:
        app.state.ready = False
        await _invoke_close(getattr(engine, "_client", None))


def _pending_limit(explicit: int | None) -> int:
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("max_pending must be a positive integer")
        return explicit
    raw = os.environ.get("RAG_MAX_PENDING", "").strip()
    if not raw:
        return DEFAULT_MAX_PENDING
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError("RAG_MAX_PENDING must be a positive integer")
    return int(raw)


def _health(engine: object, ready: bool) -> HealthResponse:
    loaded = bool(ready and getattr(engine, "loaded", False))
    num_vectors = None
    embedding_model = None
    if loaded:
        total = getattr(getattr(engine, "_index", None), "ntotal", None)
        if isinstance(total, int) and not isinstance(total, bool):
            num_vectors = total
        manifest = getattr(engine, "_manifest", None)
        model = manifest.get("model") if isinstance(manifest, dict) else None
        model_id = model.get("id") if isinstance(model, dict) else None
        if isinstance(model_id, str):
            embedding_model = model_id
    config = getattr(engine, "config", None)
    generation_model = getattr(config, "generation_model", None)
    top_k = getattr(config, "top_k", None)
    device = getattr(config, "device", None)
    if not isinstance(generation_model, str):
        generation_model = None
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        top_k = None
    if not isinstance(device, str):
        device = None
    return HealthResponse(
        status="ok" if loaded else "not_ready",
        index_loaded=loaded,
        num_vectors=num_vectors,
        embedding_model=embedding_model,
        generation_model=generation_model,
        top_k=top_k,
        device=device,
    )


def _response_from_payload(payload: dict, debug: bool) -> AskResponse:
    evidence = payload["evidence"]
    if not isinstance(evidence, list):
        raise TypeError("evidence must be a list")
    sources = []
    for number in payload["sources"]:
        if isinstance(number, bool) or not isinstance(number, int):
            continue
        if number < 1 or number > len(evidence):
            continue
        sources.append(_passage(evidence[number - 1], shorten=True, pubid_as_str=True))
    body: dict[str, object] = {
        "label": payload["label"],
        "sources": sources,
        "prompt_id": payload["prompt_id"],
        "latency_seconds": payload["latency_seconds"],
        "rate_limit_wait_seconds": payload["rate_limit_wait_seconds"],
        "generation_seconds": payload["generation_seconds"],
    }
    if payload.get("top1_score") is not None:
        body["top1_score"] = payload["top1_score"]
    if debug:
        body["raw_response"] = payload["raw_response"]
        body["evidence"] = [_passage(row, shorten=False) for row in evidence]
    return AskResponse.model_validate(body)


def _passage(row: dict, *, shorten: bool, pubid_as_str: bool = False) -> dict[str, object]:
    text = row["text"]
    if not isinstance(text, str):
        raise TypeError("evidence text must be a string")
    if shorten and len(text) > SOURCE_TEXT_LIMIT:
        text = text[:SOURCE_TEXT_LIMIT]
    pubid = row["pubid"]
    if pubid_as_str:
        pubid = str(pubid)
    return {
        "rank": row["rank"],
        "chunk_id": row["chunk_id"],
        "pubid": pubid,
        "score": row["score"],
        "text": text,
    }


def _gemini_status(kind: str) -> tuple[int, str, str]:
    if kind == "rate_limit":
        return 429, "rate_limit", "upstream rate limit exceeded"
    if kind == "timeout":
        return 504, "timeout", "upstream request timed out"
    return 502, "upstream", "upstream request failed"


def _error(
    status: int,
    code: str,
    detail: str,
    *,
    retry_after: str | None = None,
) -> JSONResponse:
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return JSONResponse(
        status_code=status,
        content={"error": code, "detail": detail},
        headers=headers,
    )


def _mounted_ui_app(app: FastAPI):
    for route in app.router.routes:
        if isinstance(route, Mount) and route.path == "/ui":
            return route.app
    raise RuntimeError("Gradio UI was not mounted")


async def _read_mounted_page(mounted, request: Request) -> tuple[int, bytes]:
    """Read the Gradio page as if the browser opened /ui/."""
    scope = dict(request.scope)
    scope["method"] = "GET"
    scope["path"] = "/"
    scope["raw_path"] = b"/"
    scope["query_string"] = b""
    scope["root_path"] = request.scope.get("root_path", "").rstrip("/") + "/ui"
    scope.pop("router", None)
    scope.pop("endpoint", None)
    scope.pop("route", None)
    status_code = 500
    chunks: list[bytes] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await mounted(scope, receive, send)
    return status_code, b"".join(chunks)


def ask_service(app_state: object, question: str, debug: bool) -> AskResponse:
    """Run one question through the shared admission limit and the engine.

    ``POST /ask`` and the Gradio handler both use this function. Success returns
    an ``AskResponse``. Failure raises ``AskServiceError`` and does not include
    a traceback in that exception's message.
    """
    started = time.perf_counter()
    if not getattr(app_state, "ready", False):
        _log_error(503, "not_ready", started)
        raise AskServiceError(503, "not_ready", "engine is not ready")
    admission = getattr(app_state, "admission", None)
    if admission is None or not admission.try_enter():
        _log_error(503, "queue_full", started)
        raise AskServiceError(
            503,
            "queue_full",
            "server is busy",
            retry_after=RETRY_AFTER_SECONDS,
        )
    try:
        try:
            payload = app_state.engine.answer(question)
        except ValueError:
            _log_error(422, "invalid_question", started)
            raise AskServiceError(422, "invalid_question", "question is invalid")
        except GeminiRequestError as exc:
            status, code, detail = _gemini_status(getattr(exc, "kind", "other"))
            _log_error(status, code, started)
            raise AskServiceError(status, code, detail)
        except Exception:
            logger.exception("ask status=500 error=internal")
            raise AskServiceError(500, "internal", "request failed")
        try:
            response = _response_from_payload(payload, debug)
        except (ValidationError, KeyError, TypeError, IndexError, AttributeError):
            logger.error("ask status=500 error=internal reason=response_schema")
            raise AskServiceError(500, "internal", "request failed")
        logger.info(
            "ask status=200 label=%s latency_seconds=%s rate_limit_wait_seconds=%s",
            response.label,
            response.latency_seconds,
            response.rate_limit_wait_seconds,
        )
        return response
    finally:
        admission.leave()


def _log_error(status: int, code: str, started: float) -> None:
    logger.info(
        "ask status=%s error=%s latency_seconds=%s",
        status,
        code,
        round(time.perf_counter() - started, 6),
    )


async def _invoke_close(client: object) -> None:
    if client is None:
        return
    method = getattr(client, "aclose", None)
    if not callable(method):
        method = getattr(client, "close", None)
    if not callable(method):
        return
    result = method()
    if inspect.isawaitable(result):
        await result


app = create_app()


def main() -> None:
    """Serve on loopback with one worker and without auto-reload."""
    import uvicorn

    uvicorn.run(
        "rag.api:app",
        host=LOOPBACK_HOST,
        port=8000,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
