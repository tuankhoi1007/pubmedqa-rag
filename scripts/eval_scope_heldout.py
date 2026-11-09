"""Score held-out questions at the candidate scope thresholds.

In-domain held-out rows are train indices 200 through 399. Paraphrases are
the 30 rewrites in tests/data/paraphrased.txt, drawn with seed 0 from the
first 200 train questions. Off-topic held-out lines are
tests/data/off_topic_heldout.txt and must not repeat tests/data/off_topic.txt.

Scoring uses RAGEngine.retrieve() only. answer() and Gemini are not called.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import eval_scope

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "pubmedqa-bge-small-v1"
DEFAULT_PARAPHRASE = ROOT / "tests" / "data" / "paraphrased.txt"
DEFAULT_OFF_TOPIC = ROOT / "tests" / "data" / "off_topic.txt"
DEFAULT_OFF_TOPIC_HELDOUT = ROOT / "tests" / "data" / "off_topic_heldout.txt"
DEFAULT_OUT = ROOT / "results" / "scope" / "heldout.json"
PARAPHRASE_SEED = 0
PARAPHRASE_N = 30
THRESHOLDS = (0.66, 0.68, 0.70)
SOURCE_LINE = re.compile(r"^#\s*(\d+)\s*\|\s*(.*)$")


def load_slice(artifact_dir: Path, start: int, stop: int) -> tuple[list[str], dict]:
    from datasets import load_dataset

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_block = manifest["dataset"]
    dataset = load_dataset(
        dataset_block["id"],
        dataset_block["config"],
        revision=dataset_block["requested_revision"],
    )
    split_name = dataset_block["split"]
    train = dataset[split_name]
    if stop > len(train) or start < 0 or start >= stop:
        raise RuntimeError(f"slice {start}:{stop} does not fit a train split of {len(train)}")
    selected = train.select(range(start, stop))
    questions = [str(question) for question in selected["question"]]
    return questions, {
        "id": dataset_block["id"],
        "config": dataset_block["config"],
        "split": split_name,
        "requested_revision": dataset_block["requested_revision"],
        "start": start,
        "stop_exclusive": stop,
        "order": "dataset order, no shuffle",
    }


def read_paraphrases(path: Path) -> list[dict[str, object]]:
    rows = []
    pending: dict[str, object] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = SOURCE_LINE.match(stripped)
        if match:
            pending = {"source_index": int(match.group(1)), "original": match.group(2).strip()}
            continue
        if stripped.startswith("#"):
            continue
        if pending is None:
            raise RuntimeError(f"paraphrase at line {line_number} has no source comment")
        rows.append(
            {
                "source_index": pending["source_index"],
                "original": pending["original"],
                "question": stripped,
            }
        )
        pending = None
    if pending is not None:
        raise RuntimeError("paraphrase file ends with a source comment and no rewrite")
    return rows


def expected_source_indices() -> list[int]:
    return sorted(random.Random(PARAPHRASE_SEED).sample(range(200), PARAPHRASE_N))


def rates(scores: list[float]) -> list[dict[str, float | int]]:
    total = len(scores)
    table = []
    for threshold in THRESHOLDS:
        blocked = sum(score < threshold for score in scores)
        table.append(
            {
                "threshold": threshold,
                "blocked": blocked,
                "blocked_rate": blocked / total,
            }
        )
    return table


def evaluate(artifact_dir: Path, paraphrase_path: Path, off_topic_path: Path, heldout_path: Path, out_path: Path) -> dict:
    import torch

    from rag.engine import RAGConfig, RAGEngine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to measure scope on CPU")
    paraphrases = read_paraphrases(paraphrase_path)
    expected = expected_source_indices()
    found = [int(row["source_index"]) for row in paraphrases]
    if found != expected:
        raise RuntimeError("paraphrase source indices do not match seed 0 sample of train 0-199")
    off_topic = eval_scope.read_off_topic(off_topic_path)
    off_heldout = eval_scope.read_off_topic(heldout_path)
    overlap = sorted(set(off_topic) & set(off_heldout))
    if overlap:
        raise RuntimeError(f"off-topic held-out repeats the tuning set: {overlap}")

    heldout_questions, dataset = load_slice(artifact_dir, 200, 400)
    originals, _ = load_slice(artifact_dir, 0, 200)
    for row in paraphrases:
        source = originals[int(row["source_index"])]
        if row["original"] != source:
            raise RuntimeError(
                f"paraphrase comment for index {row['source_index']} does not match the dataset question"
            )

    engine = RAGEngine(
        RAGConfig(artifact_dir=artifact_dir, device="cuda"),
        client=eval_scope._RefuseGemini(),
    )
    engine.load_resources()
    in_rows = eval_scope.top1_scores(engine, heldout_questions, "in-domain-heldout")
    for offset, row in enumerate(in_rows):
        row["index"] = 200 + offset
    scored_paraphrases = eval_scope.top1_scores(
        engine,
        [str(item["question"]) for item in paraphrases],
        "paraphrase",
    )
    paraphrase_rows = [
        {
            "source_index": item["source_index"],
            "original": item["original"],
            "question": item["question"],
            "top1_score": scored["top1_score"],
        }
        for item, scored in zip(paraphrases, scored_paraphrases)
    ]
    off_rows = eval_scope.top1_scores(engine, off_heldout, "off-topic-heldout")

    def pack(name: str, description: str, rows: list[dict]) -> dict:
        scores = [float(row["top1_score"]) for row in rows]
        return {
            "name": name,
            "description": description,
            "n": len(rows),
            "top1": eval_scope.score_summary(scores),
            "thresholds": rates(scores),
            "rows": rows,
        }

    summary = {
        "artifact_dir": str(artifact_dir),
        "device": "cuda",
        "method": "RAGEngine.retrieve top-1 inner product",
        "gemini_called": False,
        "block_rule": "top-1 score < threshold",
        "percentile_method": "numpy.percentile linear",
        "dataset": dataset,
        "paraphrase": {
            "path": str(paraphrase_path),
            "seed": PARAPHRASE_SEED,
            "sample": "30 indices drawn without replacement from train 0-199, then sorted",
            "source_indices": expected,
        },
        "off_topic_heldout_path": str(heldout_path),
        "thresholds_checked": list(THRESHOLDS),
        "groups": {
            "in_domain_heldout": pack(
                "in_domain_heldout",
                "train indices 200-399, unseen while choosing the threshold",
                in_rows,
            ),
            "paraphrase": pack(
                "paraphrase",
                "rewrites of 30 seed-0 questions from train 0-199",
                paraphrase_rows,
            ),
            "off_topic_heldout": pack(
                "off_topic_heldout",
                "30 new off-topic questions, disjoint from the 50 tuning questions",
                off_rows,
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    _print_report(summary, out_path)
    return summary


def _print_report(summary: dict, out_path: Path) -> None:
    print(f"gemini_called {summary['gemini_called']}")
    header = "group  n  " + "  ".join(f"{threshold:.2f}" for threshold in THRESHOLDS)
    print(header)
    for group in summary["groups"].values():
        cells = "  ".join(
            f"{row['blocked_rate']:.1%} ({row['blocked']})" for row in group["thresholds"]
        )
        print(f"{group['name']}  n={group['n']}  {cells}")
    print(out_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out scope scores. Does not call Gemini.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--paraphrase", type=Path, default=DEFAULT_PARAPHRASE)
    parser.add_argument("--off-topic", type=Path, default=DEFAULT_OFF_TOPIC)
    parser.add_argument("--off-topic-heldout", type=Path, default=DEFAULT_OFF_TOPIC_HELDOUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate(
        args.artifact_dir,
        args.paraphrase,
        args.off_topic,
        args.off_topic_heldout,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
