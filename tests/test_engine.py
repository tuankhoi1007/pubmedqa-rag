import json

import numpy as np
import pytest

from rag.engine import RAGConfig, RAGEngine


class FixedEmbedder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        row = self.vector.reshape(1, -1)
        return np.repeat(row, len(texts), axis=0)


def _write_artifact(directory, vectors, rows, **manifest_overrides):
    import faiss

    directory.mkdir(parents=True, exist_ok=True)
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(directory / "index.faiss"))
    with (directory / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "model": {
            "id": "BAAI/bge-small-en-v1.5",
            "requested_revision": "model-sha",
        },
        "pooling": "cls",
        "normalization": "l2",
        "metric": "inner_product",
        "index": "IndexFlatIP",
        "dtype": "float32",
        "dimension": int(matrix.shape[1]),
        "num_chunks": len(rows),
    }
    manifest.update(manifest_overrides)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rows():
    return [
        {"chunk_id": 0, "pubid": 111, "text": "alpha"},
        {"chunk_id": 1, "pubid": 222, "text": "beta"},
        {"chunk_id": 2, "pubid": 111, "text": "gamma"},
    ]


def _engine(directory, top_k=2):
    return RAGEngine(RAGConfig(artifact_dir=directory, device="cpu", top_k=top_k))


def test_config_stores_artifact_dir_device_and_default_top_k(tmp_path):
    config = RAGConfig(artifact_dir=tmp_path, device=" cuda ")
    engine = RAGEngine(config)

    assert engine.config is config
    assert engine.config.artifact_dir == tmp_path
    assert engine.config.device == "cuda"
    assert engine.config.top_k == 10
    assert engine.loaded is False


def test_config_rejects_bad_top_k_and_device(tmp_path):
    with pytest.raises(ValueError):
        RAGConfig(artifact_dir=tmp_path, device="", top_k=10)
    with pytest.raises(ValueError):
        RAGConfig(artifact_dir=tmp_path, device="cpu", top_k=0)
    with pytest.raises(ValueError):
        RAGConfig(artifact_dir=tmp_path, device="cpu", top_k=True)
    with pytest.raises(TypeError):
        RAGEngine({"artifact_dir": tmp_path})


def test_retrieve_before_load_and_answer_are_explicit(tmp_path):
    class UnusedClient:
        def generate(self, prompt, model):
            raise AssertionError("generate called")

    engine = RAGEngine(
        RAGConfig(artifact_dir=tmp_path, device="cpu", top_k=2),
        client=UnusedClient(),
    )

    with pytest.raises(RuntimeError, match="load_resources"):
        engine.retrieve("lace plant")
    with pytest.raises(RuntimeError, match="load_resources"):
        engine.answer("lace plant")


def test_retrieve_returns_inner_product_scores_for_chunks(tmp_path):
    _write_artifact(tmp_path, np.eye(3), _rows())
    embedder = FixedEmbedder([0.0, 0.2, 0.9])
    engine = _engine(tmp_path, top_k=2)
    engine.load_resources(embedder=embedder)

    evidence = engine.retrieve("  lace plant  ")

    assert embedder.calls == [["  lace plant  "]]
    assert [(item.rank, item.chunk_id, item.pubid, item.text) for item in evidence] == [
        (1, 2, 111, "gamma"),
        (2, 1, 222, "beta"),
    ]
    assert evidence[0].score == pytest.approx(0.9)
    assert evidence[1].score == pytest.approx(0.2)
    assert evidence[0].score + evidence[1].score == pytest.approx(1.1)


def test_second_load_keeps_the_first_resources(tmp_path):
    _write_artifact(tmp_path, np.eye(3), _rows())
    first = FixedEmbedder([0.0, 0.0, 1.0])
    second = FixedEmbedder([1.0, 0.0, 0.0])
    engine = _engine(tmp_path, top_k=1)
    engine.load_resources(embedder=first)
    index = engine._index
    engine.load_resources(embedder=second)

    evidence = engine.retrieve("gamma")

    assert engine._embedder is first
    assert engine._index is index
    assert second.calls == []
    assert evidence[0].chunk_id == 2
    assert evidence[0].text == "gamma"


def test_empty_question_is_rejected(tmp_path):
    _write_artifact(tmp_path, np.eye(3), _rows())
    embedder = FixedEmbedder([1.0, 0.0, 0.0])
    engine = _engine(tmp_path, top_k=1)
    engine.load_resources(embedder=embedder)

    with pytest.raises(ValueError, match="non-empty"):
        engine.retrieve("   ")
    assert embedder.calls == []


def test_artifact_checks_run_before_serving(tmp_path):
    vectors = np.eye(3)
    rows = _rows()
    _write_artifact(tmp_path, vectors, rows)
    embedder = FixedEmbedder([1.0, 0.0, 0.0])

    broken_ids = [dict(row) for row in rows]
    broken_ids[1]["chunk_id"] = 0
    _write_artifact(tmp_path / "ids", vectors, broken_ids)
    id_engine = _engine(tmp_path / "ids", top_k=1)
    with pytest.raises(RuntimeError, match="chunk_id"):
        id_engine.load_resources(embedder=embedder)
    assert id_engine.loaded is False

    _write_artifact(tmp_path / "dim", vectors, rows, dimension=4)
    with pytest.raises(RuntimeError, match="dimension"):
        _engine(tmp_path / "dim", top_k=1).load_resources(embedder=embedder)

    _write_artifact(tmp_path / "count", vectors, rows, num_chunks=4)
    with pytest.raises(RuntimeError, match="ntotal"):
        _engine(tmp_path / "count", top_k=1).load_resources(embedder=embedder)

    with pytest.raises(RuntimeError, match="top_k"):
        _engine(tmp_path, top_k=4).load_resources(embedder=embedder)

    _write_artifact(tmp_path / "pool", vectors, rows, pooling="mean")
    with pytest.raises(RuntimeError, match="pooling"):
        _engine(tmp_path / "pool", top_k=1).load_resources(embedder=embedder)
    assert embedder.calls == []


def test_manifest_revision_must_match_an_observed_commit(tmp_path):
    _write_artifact(
        tmp_path,
        np.eye(3),
        _rows(),
        model={
            "id": "BAAI/bge-small-en-v1.5",
            "requested_revision": "model-sha",
            "observed_commit_hash": "other-sha",
        },
    )
    engine = _engine(tmp_path, top_k=1)
    with pytest.raises(RuntimeError, match="observed commit"):
        engine.load_resources(embedder=FixedEmbedder([1.0, 0.0, 0.0]))
    assert engine.loaded is False


def test_query_dimension_must_match_the_index(tmp_path):
    _write_artifact(tmp_path, np.eye(3), _rows())
    engine = _engine(tmp_path, top_k=1)
    engine.load_resources(embedder=FixedEmbedder([1.0, 0.0]))

    with pytest.raises(RuntimeError, match="dimension"):
        engine.retrieve("lace plant")
