import json

import numpy as np
import pytest

from rag.engine import RAGConfig, RAGEngine
from rag.gemini import DEFAULT_GENERATION_MODEL, MIN_START_INTERVAL_SECONDS, GeminiClient

LACE_QUESTION = (
    "Do mitochondria play a role in remodelling lace plant leaves "
    "during programmed cell death?"
)


class ScriptedClient:
    def __init__(self, raw):
        self.raw = raw
        self.prompts = []
        self.models = []

    def generate(self, prompt, model):
        self.prompts.append(prompt)
        self.models.append(model)
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


class RoutingEmbedder:
    def embed(self, texts):
        question = texts[0]
        if question.startswith("Is metformin"):
            vector = [1.0, 0.4, 0.0]
        elif question.startswith("Does aspirin"):
            vector = [0.0, 1.0, 0.4]
        else:
            vector = [0.0, 0.0, 1.0]
        return np.asarray([vector], dtype=np.float32)


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


def _engine(directory, client, top_k=2, **config):
    _write_artifact(directory)
    engine = RAGEngine(
        RAGConfig(artifact_dir=directory, device="cpu", top_k=top_k, **config),
        client=client,
    )
    engine.load_resources(embedder=RoutingEmbedder())
    return engine


def test_two_questions_fill_their_own_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = ScriptedClient("Conclusion: seen.\nAnswer: yes\nSources: 1")
    engine = _engine(tmp_path, client, top_k=1)
    first_question = "Is metformin associated with lower mortality?"
    second_question = "Does aspirin reduce recurrent stroke?"

    first = engine.answer(first_question)
    second = engine.answer(second_question)

    assert client.models == [DEFAULT_GENERATION_MODEL, DEFAULT_GENERATION_MODEL]
    assert "Question: " + first_question in client.prompts[0]
    assert "[1] (pubid 111) alpha passage" in client.prompts[0]
    assert second_question not in client.prompts[0]
    assert "beta passage" not in client.prompts[0]
    assert "Question: " + second_question in client.prompts[1]
    assert "[1] (pubid 222) beta passage" in client.prompts[1]
    assert first_question not in client.prompts[1]
    assert "alpha passage" not in client.prompts[1]
    assert LACE_QUESTION not in client.prompts[0]
    assert LACE_QUESTION not in client.prompts[1]
    assert first["label"] == "yes"
    assert second["label"] == "yes"
    assert first["prompt_id"] == second["prompt_id"] == "exp10"
    assert first["evidence"][0]["pubid"] == 111
    assert second["evidence"][0]["pubid"] == 222
    assert first["sources"] == [1]
    assert isinstance(first["latency_seconds"], float)
    assert first["latency_seconds"] >= first["generation_seconds"] >= 0
    assert first["rate_limit_wait_seconds"] == 0.0
    assert second["rate_limit_wait_seconds"] == 0.0


def test_label_and_sources_follow_the_response(tmp_path):
    raw = "Conclusion: No difference.\nAnswer: no\nSources: 1, 9, 21645374"
    client = ScriptedClient(raw)
    engine = _engine(tmp_path, client)

    result = engine.answer("Is metformin associated with lower mortality?")

    assert result["label"] == "no"
    assert result["raw_response"] == raw
    assert result["sources"] == [1]
    assert [item["pubid"] for item in result["evidence"]] == [111, 222]
    assert 21645374 not in result["sources"]
    assert 9 not in result["sources"]


def test_empty_response_is_parse_fail(tmp_path):
    engine = _engine(tmp_path, ScriptedClient(""))

    result = engine.answer("Does aspirin reduce recurrent stroke?")

    assert result["label"] == "parse_fail"
    assert result["raw_response"] == ""
    assert result["sources"] == []
    assert result["evidence"]


def test_api_errors_are_not_labels(tmp_path):
    timeout_engine = _engine(tmp_path, ScriptedClient(TimeoutError("deadline exceeded")))
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        timeout_engine.answer("Is metformin associated with lower mortality?")

    limited = _engine(
        tmp_path,
        ScriptedClient(RuntimeError("429 RESOURCE_EXHAUSTED")),
    )
    with pytest.raises(RuntimeError, match="429 RESOURCE_EXHAUSTED"):
        limited.answer("Does aspirin reduce recurrent stroke?")


def test_generation_model_comes_from_config(tmp_path):
    client = ScriptedClient("Answer: maybe\nSources: 2")
    engine = _engine(tmp_path, client, generation_model="gemini-test-model")

    result = engine.answer("Is metformin associated with lower mortality?")

    assert client.models == ["gemini-test-model"]
    assert result["label"] == "maybe"
    assert result["sources"] == [2]


def test_answer_reports_wait_generation_and_total_latency(tmp_path, fake_gemini_clock):
    class Models:
        def __init__(self):
            self.kwargs = []

        def generate_content(self, **kwargs):
            self.kwargs.append(kwargs)
            fake_gemini_clock.advance(0.25)
            return type("Response", (), {"text": "Answer: yes\nSources: 1"})()

    class SDK:
        def __init__(self):
            self.models = Models()

    sdk = SDK()
    engine = _engine(
        tmp_path,
        GeminiClient(api_key="local-test-value", sdk_client=sdk),
        top_k=1,
    )
    first = engine.answer("Is metformin associated with lower mortality?")
    second = engine.answer("Does aspirin reduce recurrent stroke?")

    assert first["rate_limit_wait_seconds"] == 0
    assert first["generation_seconds"] == pytest.approx(0.25)
    assert first["latency_seconds"] >= 0
    assert first["label"] == "yes"
    assert "Is metformin associated with lower mortality?" in sdk.models.kwargs[0]["contents"]
    assert "temperature" not in sdk.models.kwargs[0]
    assert second["rate_limit_wait_seconds"] == pytest.approx(MIN_START_INTERVAL_SECONDS - 0.25)
    assert second["generation_seconds"] == pytest.approx(0.25)
    assert "Does aspirin reduce recurrent stroke?" in sdk.models.kwargs[1]["contents"]
    assert "lace plant" not in sdk.models.kwargs[1]["contents"]
