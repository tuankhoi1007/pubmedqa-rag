"""Measure top-1 retrieval scores for in-domain and off-topic questions.

In-domain questions are the first 200 rows of PubMedQA pqa_labeled train,
in dataset order, with no shuffle. That is the same slice as
scripts/eval_retrieval.py. Off-topic questions are the lines in
tests/data/off_topic.txt.

Each question is scored with RAGEngine.retrieve(). The recorded value is
the inner product of the top hit. This script does not call answer(),
does not call Gemini, and does not pass the rate limiter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "pubmedqa-bge-small-v1"
DEFAULT_OFF_TOPIC = ROOT / "tests" / "data" / "off_topic.txt"
DEFAULT_OUT = ROOT / "results" / "scope" / "summary.json"
IN_DOMAIN_N = 200
GRID_START = 0.40
GRID_STOP = 0.70
GRID_STEP = 0.02
MAX_IN_DOMAIN_BLOCK_RATE = 0.01


class _RefuseGemini:
    """Stand-in client. retrieve() must not call it."""

    def generate(self, prompt: str, model: str) -> str:
        raise RuntimeError("Gemini must not be called during scope measurement")


def read_off_topic(path: Path) -> list[str]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        questions.append(text)
        if not text:
            raise RuntimeError(f"empty off-topic question at line {line_number}")
    if len(questions) < 1:
        raise RuntimeError(f"no off-topic questions in {path}")
    return questions


def load_train_questions(artifact_dir: Path, n: int) -> tuple[list[str], dict]:
    from datasets import load_dataset

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
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
    if len(train) < n:
        raise RuntimeError(f"train split has {len(train)} rows, requested {n}")
    selected = train.select(range(n))
    questions = [str(question) for question in selected["question"]]
    if any(not question.strip() for question in questions):
        raise RuntimeError("a train question is empty")
    return questions, {
        "id": dataset_block["id"],
        "config": dataset_block["config"],
        "split": split_name,
        "requested_revision": dataset_block["requested_revision"],
        "n": n,
        "order": "first train rows, dataset order, no shuffle",
    }


def top1_scores(engine, questions: list[str], label: str) -> list[dict]:
    rows = []
    total = len(questions)
    for index, question in enumerate(questions):
        evidence = engine.retrieve(question)
        if not evidence or evidence[0].rank != 1:
            raise RuntimeError(f"{label} question {index} did not return rank 1")
        scores = [item.score for item in evidence]
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise RuntimeError(f"{label} question {index} scores are not in rank order")
        rows.append({"index": index, "question": question, "top1_score": scores[0]})
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == total:
            print(f"{label} {index + 1}/{total}", flush=True)
    return rows


def score_summary(scores: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(scores, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1, method="linear")),
        "p05": float(np.percentile(array, 5, method="linear")),
        "median": float(np.percentile(array, 50, method="linear")),
        "max": float(np.max(array)),
    }


def threshold_grid() -> list[float]:
    count = int(round((GRID_STOP - GRID_START) / GRID_STEP))
    return [round(GRID_START + step * GRID_STEP, 2) for step in range(count + 1)]


def block_table(in_scores: list[float], off_scores: list[float]) -> list[dict]:
    rows = []
    in_n = len(in_scores)
    off_n = len(off_scores)
    for threshold in threshold_grid():
        in_blocked = sum(score < threshold for score in in_scores)
        off_blocked = sum(score < threshold for score in off_scores)
        rows.append(
            {
                "threshold": threshold,
                "in_domain_blocked": in_blocked,
                "in_domain_blocked_rate": in_blocked / in_n,
                "off_topic_blocked": off_blocked,
                "off_topic_blocked_rate": off_blocked / off_n,
            }
        )
    return rows


def propose(table: list[dict], in_rows: list[dict]) -> dict:
    eligible = [
        row for row in table if row["in_domain_blocked_rate"] <= MAX_IN_DOMAIN_BLOCK_RATE
    ]
    if not eligible:
        return {
            "threshold": None,
            "reason": "no grid point keeps the in-domain block rate at or below 1%",
        }
    chosen = max(eligible, key=lambda row: row["threshold"])
    blocked = [
        {"index": row["index"], "question": row["question"], "top1_score": row["top1_score"]}
        for row in in_rows
        if row["top1_score"] < chosen["threshold"]
    ]
    ordered = sorted(row["top1_score"] for row in in_rows)
    keep = int(math.floor(MAX_IN_DOMAIN_BLOCK_RATE * len(ordered) + 1e-12))
    return {
        "rule": "highest grid threshold whose in-domain block rate is at most 1%",
        "block_rule": "top-1 score < threshold",
        "threshold": chosen["threshold"],
        "in_domain_blocked": chosen["in_domain_blocked"],
        "in_domain_blocked_rate": chosen["in_domain_blocked_rate"],
        "off_topic_blocked": chosen["off_topic_blocked"],
        "off_topic_blocked_rate": chosen["off_topic_blocked_rate"],
        "in_domain_blocked_questions": blocked,
        "highest_threshold_with_in_domain_block_rate_at_most_0.01": ordered[keep],
    }


def evaluate(artifact_dir: Path, off_topic_path: Path, out_path: Path) -> dict:
    import torch

    from rag.engine import RAGConfig, RAGEngine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to measure scope on CPU")
    in_questions, dataset = load_train_questions(artifact_dir, IN_DOMAIN_N)
    off_questions = read_off_topic(off_topic_path)
    engine = RAGEngine(
        RAGConfig(artifact_dir=artifact_dir, device="cuda"),
        client=_RefuseGemini(),
    )
    engine.load_resources()
    in_rows = top1_scores(engine, in_questions, "in-domain")
    off_rows = top1_scores(engine, off_questions, "off-topic")
    in_scores = [row["top1_score"] for row in in_rows]
    off_scores = [row["top1_score"] for row in off_rows]
    table = block_table(in_scores, off_scores)
    summary = {
        "artifact_dir": str(artifact_dir),
        "device": "cuda",
        "method": "RAGEngine.retrieve top-1 inner product",
        "gemini_called": False,
        "percentile_method": "numpy.percentile linear",
        "block_rule": "top-1 score < threshold",
        "dataset": dataset,
        "off_topic_path": str(off_topic_path),
        "in_domain": {
            "n": len(in_rows),
            "top1": score_summary(in_scores),
            "rows": in_rows,
        },
        "off_topic": {
            "n": len(off_rows),
            "top1": score_summary(off_scores),
            "rows": off_rows,
        },
        "thresholds": table,
        "proposal": propose(table, in_rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(out_path)
    _print_report(summary, out_path)
    return summary


def _print_report(summary: dict, out_path: Path) -> None:
    print(f"gemini_called {summary['gemini_called']}")
    for name in ("in_domain", "off_topic"):
        stats = summary[name]["top1"]
        print(
            f"{name} n={summary[name]['n']} "
            f"min={stats['min']:.4f} p01={stats['p01']:.4f} "
            f"p05={stats['p05']:.4f} median={stats['median']:.4f} max={stats['max']:.4f}"
        )
    print("threshold  in_blocked  off_blocked")
    for row in summary["thresholds"]:
        print(
            f"{row['threshold']:.2f}  "
            f"{row['in_domain_blocked_rate']:.1%} ({row['in_domain_blocked']})  "
            f"{row['off_topic_blocked_rate']:.1%} ({row['off_topic_blocked']})"
        )
    proposal = summary["proposal"]
    print(
        "proposed "
        f"{proposal.get('threshold')} "
        f"in_blocked={proposal.get('in_domain_blocked_rate')} "
        f"off_blocked={proposal.get('off_topic_blocked_rate')}"
    )
    print(out_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure top-1 retrieval scores. Does not call Gemini."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--off-topic", type=Path, default=DEFAULT_OFF_TOPIC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate(args.artifact_dir, args.off_topic, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
