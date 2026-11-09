import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_index


def test_embed_texts_is_the_shared_encoder():
    from rag.embedding import embed_texts

    assert build_index.embed_texts is embed_texts


def test_chunk_order_is_train_then_context():
    examples = [
        {"pubid": 10, "context": {"contexts": ["a", "b"]}},
        {"pubid": 20, "context": {"contexts": ["c"]}},
        {"pubid": 10, "context": {"contexts": ["", "e"]}},
    ]
    assert list(build_index.iter_chunks(examples)) == [
        (10, "a"),
        (10, "b"),
        (20, "c"),
        (10, ""),
        (10, "e"),
    ]


def _manifest(dimension: int, num_chunks: int) -> dict:
    return build_index.assemble_manifest(
        dataset_revision="fixture-dataset-revision",
        model_revision="fixture-model-revision",
        dimension=dimension,
        num_chunks=num_chunks,
        truncation={
            "padding": True,
            "truncation": True,
            "max_length_argument": None,
            "model_max_length": 512,
            "effective_max_length": 512,
        },
        library_versions={"numpy": "fixture"},
    )


def _records():
    return [
        {"pubid": 111, "text": "alpha"},
        {"pubid": 222, "text": "beta"},
        {"pubid": 111, "text": "gamma"},
    ]


def _vectors():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_save_load_maps_ids_and_search(tmp_path):
    import faiss

    records = _records()
    vectors = _vectors()
    manifest = _manifest(vectors.shape[1], len(records))
    written = build_index.save_artifacts(tmp_path, vectors, records, manifest)

    index = build_index.load_index(written["index.faiss"])
    rows = build_index.read_metadata(written["metadata.jsonl"])
    stored = json_manifest(written["manifest.json"])

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert index.ntotal == 3
    assert len(rows) == 3
    assert stored["num_chunks"] == 3
    assert stored["dimension"] == 3
    assert stored["dtype"] == "float32"
    assert stored["dataset"]["id"] == "qiaojin/PubMedQA"
    assert stored["dataset"]["config"] == "pqa_labeled"
    assert stored["dataset"]["requested_revision"] == "fixture-dataset-revision"
    assert "observed_commit_hash" not in stored["dataset"]
    assert stored["model"]["id"] == "BAAI/bge-small-en-v1.5"
    assert stored["model"]["requested_revision"] == "fixture-model-revision"
    assert "observed_commit_hash" not in stored["model"]
    assert [row["chunk_id"] for row in rows] == [0, 1, 2]
    assert [(row["pubid"], row["text"]) for row in rows] == [
        (111, "alpha"),
        (222, "beta"),
        (111, "gamma"),
    ]
    assert index.reconstruct(0).dtype == np.float32
    assert index.reconstruct(0).tolist() == pytest.approx([1.0, 0.0, 0.0])

    scores, ids = build_index.search_index(
        index, np.array([0.0, 0.2, 0.9], dtype=np.float32), k=2
    )
    assert ids[0].tolist() == [2, 1]
    assert scores[0, 0] == pytest.approx(0.9)
    hit = int(ids[0, 0])
    assert rows[hit]["chunk_id"] == hit
    assert rows[hit]["pubid"] == 111
    assert rows[hit]["text"] == "gamma"


def test_existing_artifacts_are_not_overwritten_without_force(tmp_path):
    records = _records()
    vectors = _vectors()
    manifest = _manifest(vectors.shape[1], len(records))
    build_index.save_artifacts(tmp_path, vectors, records, manifest)
    before = (tmp_path / "metadata.jsonl").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_index.save_artifacts(tmp_path, vectors, records, manifest)

    assert (tmp_path / "metadata.jsonl").read_text(encoding="utf-8") == before
    index = build_index.load_index(tmp_path / "index.faiss")
    _, ids = build_index.search_index(index, np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert int(ids[0, 0]) == 0


def test_force_replaces_artifacts(tmp_path):
    records = _records()
    build_index.save_artifacts(
        tmp_path, _vectors(), records, _manifest(3, len(records))
    )
    replaced = [
        {"pubid": 9, "text": "only"},
        {"pubid": 8, "text": "other"},
    ]
    vectors = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    build_index.save_artifacts(
        tmp_path, vectors, replaced, _manifest(2, 2), force=True
    )

    rows = build_index.read_metadata(tmp_path / "metadata.jsonl")
    assert rows == [
        {"chunk_id": 0, "pubid": 9, "text": "only"},
        {"chunk_id": 1, "pubid": 8, "text": "other"},
    ]
    index = build_index.load_index(tmp_path / "index.faiss")
    assert index.ntotal == 2
    _, ids = build_index.search_index(index, np.array([0.0, 1.0], dtype=np.float32), k=1)
    assert int(ids[0, 0]) == 0
    assert build_index.read_metadata(tmp_path / "metadata.jsonl")[0]["text"] == "only"


def test_partial_artifact_blocks_write(tmp_path):
    (tmp_path / "metadata.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_index.save_artifacts(
            tmp_path, _vectors(), _records(), _manifest(3, 3)
        )
    assert not (tmp_path / "index.faiss").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_records_observed_commit_when_present():
    manifest = build_index.assemble_manifest(
        dataset_revision="dataset-sha",
        model_revision="model-sha",
        dataset_observed_commit="dataset-sha",
        model_observed_commit="model-sha",
        dimension=3,
        num_chunks=1,
        truncation={},
        library_versions={},
    )
    assert manifest["dataset"]["requested_revision"] == "dataset-sha"
    assert manifest["dataset"]["observed_commit_hash"] == "dataset-sha"
    assert manifest["model"]["observed_commit_hash"] == "model-sha"


def test_observed_commit_must_match_requested_sha():
    assert build_index.accept_observed_commit("model", "abc", None) is None
    assert build_index.accept_observed_commit("model", "abc", "abc") == "abc"
    with pytest.raises(RuntimeError):
        build_index.accept_observed_commit("dataset", "abc", "def")


def test_dataset_fingerprint_is_not_used_as_commit():
    class Loaded:
        _fingerprint = "fingerprint-is-not-a-commit"
        fingerprint = "fingerprint-is-not-a-commit"

        class info:
            fingerprint = "also-not-a-commit"

    assert build_index.dataset_observed_commit(Loaded()) is None
    assert build_index.optional_commit_hash(object()) is None


def test_manifest_rejects_missing_revision():
    with pytest.raises(ValueError):
        build_index.assemble_manifest(
            dataset_revision="",
            model_revision="abc",
            dimension=3,
            num_chunks=1,
            truncation={},
            library_versions={},
        )


def json_manifest(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
