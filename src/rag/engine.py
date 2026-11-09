"""Dense retrieval and exp10 generation for the PubMedQA index.

``retrieve`` encodes with :func:`rag.embedding.embed_texts` and reads the
on-disk ``IndexFlatIP`` artifact. Scores are inner products of L2-normalized
vectors, not confidence probabilities. ``answer`` fills the exp10 prompt for
the question it was given and calls an injected Gemini client.
Not medical advice.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from rag.core import build_context, parse_label
from rag.embedding import (
    DTYPE,
    INDEX_TYPE,
    METRIC,
    MODEL_ID,
    NORMALIZATION,
    POOLING,
    QUERY_BATCH_SIZE,
    TextEmbedder,
)
from rag.gemini import (
    DEFAULT_GENERATION_MODEL,
    GenerationResult,
    gemini_client_from_environment,
)
from rag.prompts import PROMPT_ID, accepted_sources, render_prompt


DEFAULT_MIN_SCORE = 0.68


def _resolve_min_score(value: float | None) -> float:
    """Use an explicit score, otherwise ``RAG_MIN_SCORE``, otherwise 0.68."""
    if value is None:
        raw = os.environ.get("RAG_MIN_SCORE", "").strip()
        value = DEFAULT_MIN_SCORE if not raw else raw
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError("RAG_MIN_SCORE must be a finite number") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("min_score must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("min_score must be a finite number")
    return value


@dataclass(frozen=True)
class RAGConfig:
    """Where the index lives and how many chunks to return."""

    artifact_dir: Path
    device: str
    top_k: int = 10
    generation_model: str = DEFAULT_GENERATION_MODEL
    prompt_id: str = PROMPT_ID
    min_score: float | None = None

    def __post_init__(self) -> None:
        artifact_dir = Path(self.artifact_dir)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(self.generation_model, str) or not self.generation_model.strip():
            raise ValueError("generation_model must be a non-empty string")
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "device", self.device.strip())
        object.__setattr__(self, "generation_model", self.generation_model.strip())
        object.__setattr__(self, "prompt_id", self.prompt_id.strip())
        object.__setattr__(self, "min_score", _resolve_min_score(self.min_score))


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk.

    ``score`` is the index inner product. It is not a probability.
    """

    rank: int
    chunk_id: int
    pubid: int | str
    text: str
    score: float


class RAGEngine:
    """Retrieve PubMed passages and ask the configured generator."""

    def __init__(self, config: RAGConfig, client=None) -> None:
        if not isinstance(config, RAGConfig):
            raise TypeError("config must be a RAGConfig")
        if client is not None and not callable(getattr(client, "generate", None)):
            raise TypeError("client must provide generate(prompt, model)")
        self.config = config
        self._client = client
        self._loaded = False
        self._index = None
        self._rows: list[dict] | None = None
        self._manifest: dict | None = None
        self._embedder = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load_resources(self, embedder=None) -> None:
        """Read the artifact and prepare the embedder.

        The default embedder is ``BAAI/bge-small-en-v1.5`` at the manifest
        revision, on ``config.device``. Pass ``embedder`` only to substitute
        that load; artifact checks still run. A second call keeps the first
        resources.
        """
        if self._loaded:
            return
        index, rows, manifest = self._read_artifact()
        if embedder is None:
            embedder = self._load_manifest_embedder(manifest)
        elif not callable(getattr(embedder, "embed", None)):
            raise TypeError("embedder must provide embed(texts)")
        self._index = index
        self._rows = rows
        self._manifest = manifest
        self._embedder = embedder
        self._loaded = True

    def retrieve(self, question: str) -> list[Evidence]:
        """Return the top chunks for ``question`` in index order.

        Raises:
            RuntimeError: ``load_resources`` has not completed.
            ValueError: ``question`` is empty or only whitespace.
        """
        if not self._loaded:
            raise RuntimeError("RAGEngine.load_resources() has not been called")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        rows = self._rows
        index = self._index
        embedder = self._embedder
        if rows is None or index is None or embedder is None:
            raise RuntimeError("RAGEngine.load_resources() has not been called")

        import numpy as np

        vectors = np.asarray(embedder.embed([question]))
        if vectors.ndim != 2 or vectors.shape != (1, index.d):
            raise RuntimeError(
                f"query embedding shape {getattr(vectors, 'shape', None)} "
                f"does not match index dimension {index.d}"
            )
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        scores, ids = index.search(matrix, self.config.top_k)
        evidence: list[Evidence] = []
        for offset, chunk_id in enumerate(ids[0].tolist()):
            chunk_id = int(chunk_id)
            if chunk_id < 0 or chunk_id >= len(rows):
                raise RuntimeError(f"search returned invalid chunk id {chunk_id}")
            row = rows[chunk_id]
            evidence.append(
                Evidence(
                    rank=offset + 1,
                    chunk_id=chunk_id,
                    pubid=row["pubid"],
                    text=row["text"],
                    score=float(scores[0, offset]),
                )
            )
        return evidence

    def answer(self, question: str) -> dict[str, object]:
        """Retrieve evidence and generate a label for this ``question``.

        The payload contains ``label``, ``raw_response``, ``evidence``,
        ``latency_seconds``, ``rate_limit_wait_seconds``, ``generation_seconds``,
        ``sources``, and ``prompt_id``. ``latency_seconds`` is the wall time of
        this call, including any rate-limit wait. ``sources`` lists 1-based
        passage numbers accepted from the response. A generator exception
        propagates; it is not turned into ``maybe`` or ``parse_fail``.

        When the top retrieved score is below ``config.min_score``, the label
        is ``out_of_scope``. That return has empty ``sources``, ``top1_score``,
        and ``latency_seconds``. It does not build a prompt, call Gemini, or
        acquire the rate limiter. A score equal to ``min_score`` is in scope.

        Raises:
            RuntimeError: ``load_resources`` has not completed.
            ValueError: ``question`` is empty or only whitespace.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        started = time.perf_counter()
        evidence = self.retrieve(question)
        if not evidence:
            raise RuntimeError("retrieve returned no passages")
        top1_score = float(evidence[0].score)
        if top1_score < self.config.min_score:
            return {
                "label": "out_of_scope",
                "raw_response": "",
                "evidence": _evidence_dicts(evidence),
                "latency_seconds": time.perf_counter() - started,
                "rate_limit_wait_seconds": 0.0,
                "generation_seconds": 0.0,
                "sources": [],
                "prompt_id": self.config.prompt_id,
                "top1_score": top1_score,
            }
        context = build_context([(item.pubid, item.text) for item in evidence])
        prompt = render_prompt(question, context, self.config.prompt_id)
        client = self._client
        if client is None:
            client = gemini_client_from_environment()
        call_started = time.perf_counter()
        generated = client.generate(prompt, self.config.generation_model)
        call_elapsed = time.perf_counter() - call_started
        if isinstance(generated, GenerationResult):
            raw = generated.text
            rate_limit_wait_seconds = generated.rate_limit_wait_seconds
            generation_seconds = generated.generation_seconds
        else:
            raw = generated
            rate_limit_wait_seconds = 0.0
            generation_seconds = call_elapsed
        if raw is None:
            raw = ""
        if not isinstance(raw, str):
            raise TypeError("generation client must return a string")
        return {
            "label": parse_label(raw),
            "raw_response": raw,
            "evidence": _evidence_dicts(evidence),
            "latency_seconds": time.perf_counter() - started,
            "rate_limit_wait_seconds": rate_limit_wait_seconds,
            "generation_seconds": generation_seconds,
            "sources": accepted_sources(raw, len(evidence)),
            "prompt_id": self.config.prompt_id,
        }

    def _read_artifact(self):
        import faiss

        directory = self.config.artifact_dir
        manifest_path = directory / "manifest.json"
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.jsonl"
        for path in (manifest_path, index_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _check_manifest(manifest)
        index = faiss.read_index(str(index_path))
        rows = _read_metadata(metadata_path)
        _check_alignment(index, rows, manifest, self.config.top_k)
        return index, rows, manifest

    def _load_manifest_embedder(self, manifest: dict) -> TextEmbedder:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device_name = self.config.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device_name!r} was requested but CUDA is not available"
            )
        device = torch.device(device_name)
        model_id = manifest["model"]["id"]
        revision = manifest["model"]["requested_revision"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision)
        model.to(device)
        parameter = next(model.parameters())
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"model dtype is {parameter.dtype}, expected float32")
        if parameter.device.type != device.type:
            raise RuntimeError(f"model parameter is on {parameter.device}, expected {device}")
        if device.index is not None and parameter.device.index != device.index:
            raise RuntimeError(f"model parameter is on {parameter.device}, expected {device}")
        return TextEmbedder(tokenizer, model, device, batch_size=QUERY_BATCH_SIZE)


def _evidence_dicts(evidence: list[Evidence]) -> list[dict[str, object]]:
    return [
        {
            "rank": item.rank,
            "chunk_id": item.chunk_id,
            "pubid": item.pubid,
            "text": item.text,
            "score": item.score,
        }
        for item in evidence
    ]


def _require_text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"manifest {label} is missing")
    return value


def _check_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest is not an object")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("manifest is missing model")
    model_id = model.get("id")
    if model_id != MODEL_ID:
        raise RuntimeError(f"manifest model.id={model_id!r}, expected {MODEL_ID!r}")
    requested = _require_text(model.get("requested_revision"), "model.requested_revision")
    observed = model.get("observed_commit_hash")
    if observed is not None and observed != requested:
        raise RuntimeError(
            f"manifest model observed commit {observed} != requested revision {requested}"
        )
    expected = {
        "pooling": POOLING,
        "normalization": NORMALIZATION,
        "metric": METRIC,
        "index": INDEX_TYPE,
        "dtype": DTYPE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"manifest {key}={manifest.get(key)!r}, expected {value!r}")
    _positive_int(manifest.get("dimension"), "dimension")
    _positive_int(manifest.get("num_chunks"), "num_chunks")


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"manifest {label} must be a positive integer")
    return value


def _read_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"metadata line {line_number} is not an object")
            rows.append(row)
    return rows


def _check_alignment(index, rows: list[dict], manifest: dict, top_k: int) -> None:
    import faiss

    if not isinstance(index, faiss.IndexFlatIP):
        raise RuntimeError(f"index type is {type(index).__name__}, expected IndexFlatIP")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError(f"index metric_type={index.metric_type}, expected inner product")
    dimension = manifest["dimension"]
    num_chunks = manifest["num_chunks"]
    if index.d != dimension:
        raise RuntimeError(f"index dimension {index.d} != manifest dimension {dimension}")
    if index.ntotal != len(rows) or index.ntotal != num_chunks:
        raise RuntimeError(
            f"index ntotal={index.ntotal}, metadata lines={len(rows)}, "
            f"manifest num_chunks={num_chunks}"
        )
    if top_k > index.ntotal:
        raise RuntimeError(f"top_k {top_k} exceeds index size {index.ntotal}")
    for position, row in enumerate(rows):
        chunk_id = row.get("chunk_id")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id != position:
            raise RuntimeError(f"chunk_id breaks at row {position}")
        pubid = row.get("pubid")
        if isinstance(pubid, bool) or not isinstance(pubid, (int, str)):
            raise RuntimeError(f"metadata pubid is invalid at row {position}")
        if not isinstance(row.get("text"), str):
            raise RuntimeError(f"metadata text is invalid at row {position}")
