"""Score PubMedQA end to end through ``RAGEngine.answer``.

The held-out slice is train indices ``[start, end)``. The default is
``[200, 700)``, after the 200 rows used to measure retrieval. Each finished
answer is appended to ``per_question.jsonl`` and flushed before the next
question, so a restarted run skips indices already stored.

A ``GeminiRequestError`` is appended to ``errors.jsonl`` and is not stored
as an answer, so that index is tried again on the next run. Three
``rate_limit`` errors in a row stop the process. A timeout, another Gemini
error, or a successful answer resets that streak.

``out_of_scope`` and ``parse_fail`` stay in the accuracy denominator. Both
are wrong predictions and lower recall of the gold label. Macro-F1 is the
unweighted mean of yes, no, and maybe only. The confusion matrix still has
columns for ``parse_fail`` and ``out_of_scope``.

This module does not change retrieval, the prompt, or the Gemini adapter.
Running ``main`` calls Gemini. Tests pass an object with ``answer`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

from rag.gemini import GeminiRequestError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "pubmedqa-bge-small-v1"
DEFAULT_OUT = ROOT / "results" / "e2e-heldout-exp10-k10"
PER_QUESTION_NAME = "per_question.jsonl"
ERRORS_NAME = "errors.jsonl"
SUMMARY_NAME = "summary.json"
RATE_LIMIT_STREAK_LIMIT = 3
GOLD_LABELS = ("yes", "no", "maybe")
NON_ANSWER_LABELS = ("parse_fail", "out_of_scope")
Z_95 = NormalDist().inv_cdf(0.975)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAGEngine.answer on a PubMedQA slice. This calls Gemini."
    )
    parser.add_argument("--start", type=int, default=200, help="First train index (default 200)")
    parser.add_argument(
        "--end",
        type=int,
        default=700,
        help="Train index stop, exclusive (default 700)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for per_question.jsonl, errors.jsonl, and summary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum questions to attempt this run, including Gemini errors",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args(argv)


def check_args(args: argparse.Namespace) -> None:
    if isinstance(args.start, bool) or isinstance(args.end, bool):
        raise ValueError("require 0 <= --start < --end")
    if args.start < 0 or args.end <= args.start:
        raise ValueError("require 0 <= --start < --end")
    if args.limit is not None and (isinstance(args.limit, bool) or args.limit < 1):
        raise ValueError("--limit must be a positive integer")


def manifest_sha256(artifact_dir: Path) -> str:
    """SHA-256 hex digest of the raw ``manifest.json`` bytes."""
    path = Path(artifact_dir) / "manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(correct: int, n: int) -> dict[str, float | str]:
    """Wilson score interval for ``correct`` successes in ``n`` trials at 95%."""
    if isinstance(correct, bool) or isinstance(n, bool):
        raise ValueError("correct and n must be integers")
    if not isinstance(correct, int) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if correct < 0 or correct > n:
        raise ValueError("correct must be between 0 and n")
    z = Z_95
    proportion = correct / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * n)) / n)
        / denominator
    )
    low = center - margin
    high = center + margin
    if correct == 0:
        low = 0.0
    if correct == n:
        high = 1.0
    return {
        "method": "wilson",
        "level": 0.95,
        "low": min(max(low, 0.0), 1.0),
        "high": min(max(high, 0.0), 1.0),
    }


def _prediction_columns(preds: list[str]) -> list[str]:
    """Gold labels, then the two non-answers, then any other prediction."""
    extras = sorted(set(preds).difference(GOLD_LABELS).difference(NON_ANSWER_LABELS))
    return list(GOLD_LABELS) + list(NON_ANSWER_LABELS) + extras


def _prf(golds: list[str], preds: list[str], label: str) -> dict[str, float | int]:
    tp = sum(gold == label and pred == label for gold, pred in zip(golds, preds))
    fp = sum(gold != label and pred == label for gold, pred in zip(golds, preds))
    fn = sum(gold == label and pred != label for gold, pred in zip(golds, preds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
    }


def build_summary(rows: list[dict], config: dict) -> dict:
    """Score every stored row. Non-answers stay in ``n`` and out of macro-F1."""
    golds = [str(row["gold"]) for row in rows]
    preds = [str(row["pred"]) for row in rows]
    n = len(rows)
    correct = sum(gold == pred for gold, pred in zip(golds, preds))
    per_label = {label: _prf(golds, preds, label) for label in GOLD_LABELS} if n else {}
    row_labels = list(GOLD_LABELS)
    column_labels = _prediction_columns(preds)
    matrix = [
        [
            sum(gold == row_label and pred == column_label for gold, pred in zip(golds, preds))
            for column_label in column_labels
        ]
        for row_label in row_labels
    ]
    parse_fail = sum(pred == "parse_fail" for pred in preds)
    out_of_scope = sum(pred == "out_of_scope" for pred in preds)
    answered = [(gold, pred) for gold, pred in zip(golds, preds) if pred in GOLD_LABELS]
    answered_correct = sum(gold == pred for gold, pred in answered)
    summary = {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "accuracy_ci95": wilson_interval(correct, n) if n else None,
        "coverage": ((n - out_of_scope - parse_fail) / n) if n else None,
        "accuracy_on_answered": (answered_correct / len(answered)) if answered else None,
        "macro_f1": (
            sum(per_label[label]["f1"] for label in GOLD_LABELS) / len(GOLD_LABELS)
        ) if n else None,
        "per_label": per_label,
        "confusion_matrix": {
            "row_labels": row_labels,
            "column_labels": column_labels,
            "matrix": matrix,
        },
        "parse_fail": parse_fail,
        "out_of_scope": out_of_scope,
        "config": {
            "model": config["model"],
            "prompt_id": config["prompt_id"],
            "top_k": config["top_k"],
            "min_score": config["min_score"],
            "artifact_manifest_hash": config["artifact_manifest_hash"],
            "artifact_manifest_hash_algorithm": "sha256",
        },
        "scoring": {
            "accuracy": "pred == gold over every stored question; out_of_scope and parse_fail remain in n",
            "coverage": "(n - out_of_scope - parse_fail) / n",
            "accuracy_on_answered": "pred == gold among predictions that are yes, no, or maybe",
            "macro_f1": "unweighted mean of F1 for yes, no, and maybe; a non-answer is a false negative for its gold label",
            "confusion_matrix": "rows are gold yes/no/maybe; columns are those labels plus parse_fail and out_of_scope",
            "interval": "Wilson score interval at 95%",
            "rows": "every line in per_question.jsonl",
        },
    }
    return summary


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path} line {line_number} is not JSON") from exc
        if not isinstance(row, dict) or "index" not in row:
            raise RuntimeError(f"{path} line {line_number} is missing index")
        index = row["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError(f"{path} line {line_number} index is not an integer")
        if index in seen:
            raise RuntimeError(f"{path} duplicate index {index}")
        seen.add(index)
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_summary(per_path: Path, summary_path: Path, config: dict) -> dict:
    summary = build_summary(read_jsonl(per_path), config)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    return summary


def result_row(example: dict, result: dict) -> dict:
    label = result.get("label")
    if not isinstance(label, str) or not label:
        raise RuntimeError(f"answer() for index {example['index']} returned no label")
    if "top1_score" in result and result["top1_score"] is not None:
        top1_score = float(result["top1_score"])
    else:
        evidence = result.get("evidence") or []
        if not evidence or "score" not in evidence[0]:
            raise RuntimeError(f"answer() for index {example['index']} has no top-1 score")
        top1_score = float(evidence[0]["score"])
    sources = result.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError(f"answer() for index {example['index']} sources must be a list")
    return {
        "index": example["index"],
        "pubid": example["pubid"],
        "question": example["question"],
        "gold": example["gold"],
        "pred": label,
        "top1_score": top1_score,
        "sources": sources,
        "latency": float(result["latency_seconds"]),
        "generation_seconds": float(result["generation_seconds"]),
    }


def run_questions(examples: list[dict], engine, out_dir: Path, *, limit: int | None, config: dict) -> dict:
    """Answer pending examples and refresh ``summary.json`` from the whole file."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / PER_QUESTION_NAME
    error_path = out_dir / ERRORS_NAME
    summary_path = out_dir / SUMMARY_NAME
    done = {row["index"] for row in read_jsonl(per_path)}
    pending = [example for example in examples if example["index"] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(
        f"pending {len(pending)} already_done {len(done)} limit {limit if limit is not None else 'all'}",
        flush=True,
    )
    streak = 0
    stopped = False
    attempted = 0
    try:
        for example in pending:
            try:
                result = engine.answer(example["question"])
            except GeminiRequestError as exc:
                append_jsonl(
                    error_path,
                    {
                        "index": example["index"],
                        "pubid": example["pubid"],
                        "question": example["question"],
                        "kind": exc.kind,
                    },
                )
                attempted += 1
                print(f"error index={example['index']} kind={exc.kind}", flush=True)
                if exc.kind == "rate_limit":
                    streak += 1
                    if streak >= RATE_LIMIT_STREAK_LIMIT:
                        stopped = True
                        print(
                            f"stopped after {RATE_LIMIT_STREAK_LIMIT} consecutive rate_limit errors",
                            flush=True,
                        )
                        break
                else:
                    streak = 0
                continue
            streak = 0
            row = result_row(example, result)
            append_jsonl(per_path, row)
            attempted += 1
            print(
                f"index={row['index']} gold={row['gold']} pred={row['pred']} "
                f"top1={row['top1_score']:.4f} latency={row['latency']:.3f}",
                flush=True,
            )
    finally:
        summary = write_summary(per_path, summary_path, config)
    if summary["n"]:
        answered = summary["accuracy_on_answered"]
        answered_text = "none" if answered is None else f"{answered:.4f}"
        print(
            f"n={summary['n']} accuracy={summary['accuracy']:.4f} "
            f"coverage={summary['coverage']:.4f} accuracy_on_answered={answered_text} "
            f"macro_f1={summary['macro_f1']:.4f} "
            f"parse_fail={summary['parse_fail']} out_of_scope={summary['out_of_scope']}",
            flush=True,
        )
    else:
        print("n=0", flush=True)
    print(summary_path, flush=True)
    return {
        "stopped": stopped,
        "attempted": attempted,
        "summary": summary,
        "per_question": per_path,
        "errors": error_path,
    }


def _json_pubid(value):
    from numbers import Integral

    if isinstance(value, bool) or not isinstance(value, Integral):
        return value
    return int(value)


def load_examples(artifact_dir: Path, start: int, end: int) -> list[dict]:
    """Train rows ``[start, end)`` at the manifest dataset revision."""
    from datasets import load_dataset

    manifest = json.loads((Path(artifact_dir) / "manifest.json").read_text(encoding="utf-8"))
    dataset_block = manifest["dataset"]
    dataset = load_dataset(
        dataset_block["id"],
        dataset_block["config"],
        revision=dataset_block["requested_revision"],
    )
    split_name = dataset_block["split"]
    if split_name not in dataset:
        raise KeyError(f"split {split_name!r} is not in the loaded dataset")
    train = dataset[split_name]
    if len(train) < end:
        raise RuntimeError(f"train split has {len(train)} rows, requested end {end}")
    selected = train.select(range(start, end))
    examples = []
    for offset, row in enumerate(selected):
        index = start + offset
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"train index {index} has an empty question")
        gold = str(row["final_decision"]).strip().lower()
        if gold not in ("yes", "no", "maybe"):
            raise RuntimeError(f"train index {index} has unexpected gold {gold!r}")
        examples.append(
            {
                "index": index,
                "pubid": _json_pubid(row["pubid"]),
                "question": question,
                "gold": gold,
            }
        )
    return examples


def _ensure_api_key() -> None:
    """Load a local ``.env`` without overriding the process environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ROOT / ".env", override=False)
    from rag.gemini import require_api_key

    require_api_key(os.environ)


def evaluate(artifact_dir: Path, out_dir: Path, *, start: int, end: int, limit: int | None) -> dict:
    """Load the index and call ``RAGEngine.answer`` on the slice."""
    import torch

    from rag.engine import DEFAULT_MIN_SCORE, RAGConfig, RAGEngine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run end-to-end on CPU")
    _ensure_api_key()
    examples = load_examples(artifact_dir, start, end)
    engine = RAGEngine(
        RAGConfig(
            artifact_dir=artifact_dir,
            device="cuda",
            top_k=10,
            min_score=DEFAULT_MIN_SCORE,
        )
    )
    engine.load_resources()
    config = {
        "model": engine.config.generation_model,
        "prompt_id": engine.config.prompt_id,
        "top_k": engine.config.top_k,
        "min_score": engine.config.min_score,
        "artifact_manifest_hash": manifest_sha256(artifact_dir),
    }
    print(f"slice [{start}, {end}) model={config['model']} prompt={config['prompt_id']}", flush=True)
    return run_questions(examples, engine, out_dir, limit=limit, config=config)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        check_args(args)
    except ValueError as exc:
        print(exc, flush=True)
        return 2
    result = evaluate(
        args.artifact_dir,
        args.out_dir,
        start=args.start,
        end=args.end,
        limit=args.limit,
    )
    if result["stopped"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
