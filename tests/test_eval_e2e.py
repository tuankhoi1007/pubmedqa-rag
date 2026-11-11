import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_e2e
from rag.gemini import GeminiRequestError

CONFIG = {
    "model": "fake-model",
    "prompt_id": "exp10",
    "top_k": 10,
    "min_score": 0.68,
    "artifact_manifest_hash": "abc",
}


def _example(index: int, gold: str, question: str | None = None) -> dict:
    return {
        "index": index,
        "pubid": 1000 + index,
        "question": question or f"question {index}",
        "gold": gold,
    }


def _payload(label: str, score: float = 0.91, sources: list[int] | None = None) -> dict:
    body = {
        "label": label,
        "raw_response": f"Answer: {label}",
        "evidence": [
            {"rank": 1, "chunk_id": 0, "pubid": 1, "text": "passage", "score": score}
        ],
        "latency_seconds": 1.25,
        "rate_limit_wait_seconds": 4.1,
        "generation_seconds": 0.8,
        "sources": [1] if sources is None else sources,
        "prompt_id": "exp10",
    }
    if label == "out_of_scope":
        body["top1_score"] = score
        body["sources"] = []
        body["generation_seconds"] = 0.0
        body["rate_limit_wait_seconds"] = 0.0
    return body


class FakeEngine:
    """Stand-in for RAGEngine. ``answer`` never calls Gemini."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.questions: list[str] = []

    def answer(self, question: str):
        self.questions.append(question)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cli_defaults():
    args = eval_e2e.parse_args([])
    assert args.start == 200
    assert args.end == 700
    assert args.out_dir == eval_e2e.DEFAULT_OUT
    assert args.limit is None
    limited = eval_e2e.parse_args(["--limit", "250"])
    assert limited.limit == 250
    assert eval_e2e.main(["--start", "5", "--end", "5"]) == 2


def test_wilson_interval_matches_the_score_formula():
    both = eval_e2e.wilson_interval(2, 4)
    assert both["low"] == pytest.approx(0.15003898915214958)
    assert both["high"] == pytest.approx(0.8499610108478504)
    none = eval_e2e.wilson_interval(0, 10)
    assert none["low"] == 0.0
    assert none["high"] == pytest.approx(0.27753279986288915)
    all_correct = eval_e2e.wilson_interval(10, 10)
    assert all_correct["low"] == pytest.approx(0.7224672001371109)
    assert all_correct["high"] == 1.0


def test_manifest_hash_is_sha256_of_the_file(tmp_path):
    import hashlib

    payload = b'{"dataset": "x"}\n'
    (tmp_path / "manifest.json").write_bytes(payload)
    assert eval_e2e.manifest_sha256(tmp_path) == hashlib.sha256(payload).hexdigest()


def test_resume_skips_a_stored_index_and_flushes_before_the_next(tmp_path):
    per_path = tmp_path / "per_question.jsonl"
    per_path.write_text(
        json.dumps(
            {
                "index": 200,
                "pubid": 1200,
                "question": "already done",
                "gold": "yes",
                "pred": "yes",
                "top1_score": 0.9,
                "sources": [1],
                "latency": 0.2,
                "generation_seconds": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = [
        _example(200, "yes", "already done"),
        _example(201, "no", "first new"),
        _example(202, "maybe", "second new"),
    ]

    class CheckingEngine(FakeEngine):
        def answer(self, question: str):
            if question == "second new":
                stored = per_path.read_text(encoding="utf-8")
                assert "first new" in stored
            return super().answer(question)

    engine = CheckingEngine([_payload("no", 0.8), _payload("maybe", 0.7)])
    first = eval_e2e.run_questions(examples, engine, tmp_path, limit=None, config=CONFIG)
    second = eval_e2e.run_questions(examples, engine, tmp_path, limit=None, config=CONFIG)

    assert engine.questions == ["first new", "second new"]
    assert second["attempted"] == 0
    indexes = [row["index"] for row in _lines(first["per_question"])]
    assert indexes == [200, 201, 202]
    assert first["summary"]["n"] == 3
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["n"] == 3


def test_gemini_errors_are_left_out_of_per_question(tmp_path):
    examples = [_example(201, "yes"), _example(202, "no"), _example(203, "maybe")]
    engine = FakeEngine(
        [
            GeminiRequestError("Gemini request failed", kind="other"),
            _payload("no"),
            _payload("maybe"),
        ]
    )
    eval_e2e.run_questions(examples, engine, tmp_path, limit=2, config=CONFIG)

    stored = _lines(tmp_path / "per_question.jsonl")
    errors = _lines(tmp_path / "errors.jsonl")
    assert [row["index"] for row in stored] == [202]
    assert stored[0]["pred"] == "no"
    assert stored[0]["latency"] == pytest.approx(1.25)
    assert "latency_seconds" not in stored[0]
    assert [row["index"] for row in errors] == [201]
    assert errors[0]["kind"] == "other"
    assert errors[0]["question"] == "question 201"
    assert engine.questions == ["question 201", "question 202"]


def test_timeout_resets_the_rate_limit_streak(tmp_path):
    examples = [_example(index, "yes") for index in range(5)]
    engine = FakeEngine(
        [
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            GeminiRequestError("Gemini request failed", kind="timeout"),
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            _payload("yes"),
        ]
    )
    result = eval_e2e.run_questions(examples, engine, tmp_path, limit=None, config=CONFIG)

    assert result["stopped"] is False
    assert len(engine.questions) == 5
    assert [row["index"] for row in _lines(result["per_question"])] == [4]
    assert [row["kind"] for row in _lines(result["errors"])] == [
        "rate_limit",
        "rate_limit",
        "timeout",
        "rate_limit",
    ]


def test_three_consecutive_rate_limits_stop_before_the_next_question(tmp_path):
    examples = [_example(index, "yes") for index in range(7)]
    engine = FakeEngine(
        [
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            _payload("yes"),
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            GeminiRequestError("Gemini request failed", kind="rate_limit"),
            _payload("no"),
        ]
    )
    result = eval_e2e.run_questions(examples, engine, tmp_path, limit=None, config=CONFIG)

    assert result["stopped"] is True
    assert engine.questions == [f"question {index}" for index in range(6)]
    assert [row["index"] for row in _lines(result["per_question"])] == [2]
    assert [row["kind"] for row in _lines(result["errors"])] == ["rate_limit"] * 5
    assert result["summary"]["n"] == 1
    assert result["summary"]["accuracy"] == pytest.approx(1.0)


def test_summary_counts_out_of_scope_as_wrong(tmp_path):
    examples = [
        _example(200, "yes", "q-yes"),
        _example(201, "no", "q-no"),
        _example(202, "maybe", "q-maybe"),
        _example(203, "yes", "q-fail"),
    ]
    engine = FakeEngine(
        [
            _payload("yes", 0.91),
            _payload("out_of_scope", 0.5),
            _payload("maybe", 0.8),
            _payload("parse_fail", 0.77, sources=[]),
        ]
    )
    result = eval_e2e.run_questions(examples, engine, tmp_path, limit=None, config=CONFIG)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    stored = _lines(result["per_question"])

    assert summary["n"] == 4
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["accuracy_ci95"]["low"] == pytest.approx(0.15003898915214958)
    assert summary["accuracy_ci95"]["high"] == pytest.approx(0.8499610108478504)
    assert summary["parse_fail"] == 1
    assert summary["out_of_scope"] == 1
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["accuracy_on_answered"] == pytest.approx(1.0)
    assert summary["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.0 + 1.0) / 3.0)
    assert list(summary["per_label"]) == ["yes", "no", "maybe"]
    assert summary["per_label"]["yes"]["precision"] == pytest.approx(1.0)
    assert summary["per_label"]["yes"]["recall"] == pytest.approx(0.5)
    assert summary["per_label"]["yes"]["f1"] == pytest.approx(2.0 / 3.0)
    assert summary["per_label"]["yes"]["support"] == 2
    assert summary["per_label"]["no"]["recall"] == 0.0
    assert summary["per_label"]["no"]["f1"] == 0.0
    assert summary["per_label"]["maybe"]["f1"] == pytest.approx(1.0)
    matrix_block = summary["confusion_matrix"]
    assert matrix_block["row_labels"] == ["yes", "no", "maybe"]
    assert matrix_block["column_labels"] == ["yes", "no", "maybe", "parse_fail", "out_of_scope"]
    rows = matrix_block["row_labels"]
    columns = matrix_block["column_labels"]
    matrix = matrix_block["matrix"]
    assert matrix[rows.index("yes")][columns.index("yes")] == 1
    assert matrix[rows.index("yes")][columns.index("parse_fail")] == 1
    assert matrix[rows.index("no")][columns.index("out_of_scope")] == 1
    assert matrix[rows.index("maybe")][columns.index("maybe")] == 1
    assert summary["config"] == {
        "model": "fake-model",
        "prompt_id": "exp10",
        "top_k": 10,
        "min_score": 0.68,
        "artifact_manifest_hash": "abc",
        "artifact_manifest_hash_algorithm": "sha256",
    }
    scoped = stored[1]
    assert scoped["pred"] == "out_of_scope"
    assert scoped["sources"] == []
    assert scoped["top1_score"] == pytest.approx(0.5)
    assert scoped["generation_seconds"] == 0.0
    assert scoped["latency"] == pytest.approx(1.25)


def test_in_scope_top1_score_comes_from_evidence():
    row = eval_e2e.result_row(_example(1, "yes"), _payload("yes", 0.83))
    assert row["top1_score"] == pytest.approx(0.83)
    assert "top1_score" not in _payload("yes", 0.83)
