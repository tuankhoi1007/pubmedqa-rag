"""Score dense retrieval with the notebook hit definition.

``eval_retrieval`` in notebooks/Thực_hành_clean.ipynb walks the retrieved
chunk pubids in rank order. A question is a hit when any pubid equals the
question pubid. Recall@k is hits / n. MRR@k is the mean of 1/rank for the
first hit whose rank is within k, and 0 when that hit is absent. Duplicate
pubids stay in the ranking; they are not collapsed before scoring.

The printed dense result Recall@5 = 0.985, MRR = 0.968 is that function at
k=5 (n=200, first train rows). The same function at k=10 printed
Recall@10 = 0.990, MRR = 0.969. Those figures are recorded for comparison.
This script does not change the ranking to chase them.

The script reads an existing index. It does not rebuild it and does not
call Gemini.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from numbers import Integral
from pathlib import Path

import build_index
from rag.embedding import embed_texts

KS = (5, 10)
PER_QUESTION_NAME = "per_question.jsonl"
SUMMARY_NAME = "summary.json"

# Stdout from eval_retrieval(search) and the later k loop. Not targets.
NOTEBOOK_STDOUT = {
    5: {"recall": 0.985, "mrr": 0.968},
    10: {"recall": 0.990, "mrr": 0.969},
}


def first_hit_rank(retrieved_pubids, gold_pubid) -> int | None:
    """Return the 1-based chunk rank of the first matching pubid."""
    for offset, pubid in enumerate(retrieved_pubids):
        if pubid == gold_pubid:
            return offset + 1
    return None


def reciprocal_rank(rank: int | None, k: int) -> float:
    """Reciprocal rank inside cutoff k. Missing or past k is 0."""
    if k < 1:
        raise ValueError("k must be positive")
    if rank is not None and rank <= k:
        return 1.0 / rank
    return 0.0


def summarize_ranks(ranks: list[int | None], k: int) -> dict[str, float | int]:
    """Hit-based Recall@k and MRR@k over one rank per question."""
    if k < 1:
        raise ValueError("k must be positive")
    n = len(ranks)
    if n == 0:
        raise ValueError("no questions to score")
    hits = sum(rank is not None and rank <= k for rank in ranks)
    mrr = sum(reciprocal_rank(rank, k) for rank in ranks) / n
    return {"k": k, "n": n, "hits": hits, "recall": hits / n, "mrr": mrr}


def result_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / PER_QUESTION_NAME, out_dir / SUMMARY_NAME


def _refuse_existing(out_dir: Path) -> None:
    existing = [path for path in result_paths(out_dir) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite {joined}")


def write_results(out_dir: Path, questions: list[dict], summary: dict) -> dict[str, Path]:
    """Write the per-question JSONL and the summary. Refuse to replace them."""
    out_dir = Path(out_dir)
    per_path, summary_path = result_paths(out_dir)
    _refuse_existing(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        per_tmp = tmp_dir / PER_QUESTION_NAME
        with per_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in questions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_tmp = tmp_dir / SUMMARY_NAME
        summary_tmp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(per_tmp, per_path)
        os.replace(summary_tmp, summary_path)
    return {"per_question": per_path, "summary": summary_path}


def _require_text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"manifest {label} is missing")
    return value.strip()


def _check_manifest(manifest: dict) -> None:
    dataset = manifest.get("dataset")
    model = manifest.get("model")
    if not isinstance(dataset, dict) or not isinstance(model, dict):
        raise RuntimeError("manifest is missing dataset or model")
    expected_dataset = {
        "id": build_index.DATASET_ID,
        "config": build_index.DATASET_CONFIG,
        "split": build_index.DATASET_SPLIT,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise RuntimeError(
                f"manifest dataset.{key}={dataset.get(key)!r}, expected {expected!r}"
            )
    if model.get("id") != build_index.MODEL_ID:
        raise RuntimeError(
            f"manifest model.id={model.get('id')!r}, expected {build_index.MODEL_ID!r}"
        )
    for key, expected in (
        ("pooling", build_index.POOLING),
        ("normalization", build_index.NORMALIZATION),
        ("metric", build_index.METRIC),
        ("index", build_index.INDEX_TYPE),
        ("dtype", build_index.DTYPE),
    ):
        if manifest.get(key) != expected:
            raise RuntimeError(f"manifest {key}={manifest.get(key)!r}, expected {expected!r}")
    for label, block in (("dataset", dataset), ("model", model)):
        requested = _require_text(block.get("requested_revision"), f"{label}.requested_revision")
        observed = block.get("observed_commit_hash")
        if observed is not None and observed != requested:
            raise RuntimeError(
                f"manifest {label} observed commit {observed} != requested {requested}"
            )


def _pubids_comparable(question_pubid, metadata_pubid) -> None:
    if isinstance(question_pubid, bool) or isinstance(metadata_pubid, bool):
        raise RuntimeError("pubid is bool; refusing to score")
    same_type = type(question_pubid) is type(metadata_pubid)
    both_ints = isinstance(question_pubid, Integral) and isinstance(metadata_pubid, Integral)
    if same_type or both_ints:
        return
    raise RuntimeError(
        "question pubid type "
        f"{type(question_pubid).__name__} does not match metadata pubid type "
        f"{type(metadata_pubid).__name__}"
    )


def _json_pubid(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        return value
    return int(value)


def load_artifact(artifact_dir: Path):
    """Load index, metadata, and manifest. Stop when they disagree."""
    import faiss

    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_manifest(manifest)
    index = build_index.load_index(artifact_dir / "index.faiss")
    rows = build_index.read_metadata(artifact_dir / "metadata.jsonl")
    if not isinstance(index, faiss.IndexFlatIP):
        raise RuntimeError(f"index type is {type(index).__name__}, expected IndexFlatIP")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError(
            f"index metric_type={index.metric_type}, expected inner product"
        )
    if index.d != manifest["dimension"]:
        raise RuntimeError(
            f"index dimension {index.d} != manifest dimension {manifest['dimension']}"
        )
    if index.ntotal != len(rows) or index.ntotal != manifest["num_chunks"]:
        raise RuntimeError(
            f"index ntotal={index.ntotal}, metadata lines={len(rows)}, "
            f"manifest num_chunks={manifest['num_chunks']}"
        )
    for row_index, row in enumerate(rows):
        if row.get("chunk_id") != row_index:
            raise RuntimeError(f"chunk_id breaks at row {row_index}")
    return index, rows, manifest


def evaluate(artifact_dir: Path, out_dir: Path, *, n: int, batch_size: int) -> dict[str, Path]:
    """Encode the first n train questions and score the on-disk index."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    if n < 1:
        raise ValueError("n must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _refuse_existing(Path(out_dir))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to evaluate on CPU")

    index, rows, manifest = load_artifact(artifact_dir)
    if index.ntotal < max(KS):
        raise RuntimeError(f"index has {index.ntotal} vectors, need at least {max(KS)}")

    dataset_revision = manifest["dataset"]["requested_revision"]
    model_revision = manifest["model"]["requested_revision"]
    dataset = load_dataset(
        manifest["dataset"]["id"],
        manifest["dataset"]["config"],
        revision=dataset_revision,
    )
    split_name = manifest["dataset"]["split"]
    if split_name not in dataset:
        raise KeyError(f"split {split_name!r} is not in the loaded dataset")
    train = dataset[split_name]
    if len(train) < n:
        raise RuntimeError(f"train split has {len(train)} rows, requested {n}")
    questions = train.select(range(n))
    _pubids_comparable(questions[0]["pubid"], rows[0]["pubid"])

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["id"], revision=model_revision)
    model = AutoModel.from_pretrained(manifest["model"]["id"], revision=model_revision)
    model.to(device)
    parameter = next(model.parameters())
    if parameter.device.type != "cuda":
        raise RuntimeError(f"model parameter is on {parameter.device}, expected cuda")
    if parameter.dtype != torch.float32:
        raise RuntimeError(f"model dtype is {parameter.dtype}, expected float32")

    encode_started = time.perf_counter()
    vectors = embed_texts(
        list(questions["question"]),
        tokenizer,
        model,
        device,
        batch_size,
    )
    encode_seconds = time.perf_counter() - encode_started
    if vectors.shape != (n, index.d) or vectors.dtype != np.float32:
        raise RuntimeError(
            f"query matrix is {vectors.shape} {vectors.dtype}, expected ({n}, {index.d}) float32"
        )

    search_k = max(KS)
    search_started = time.perf_counter()
    _scores, ids = build_index.search_index(index, vectors, search_k)
    search_seconds = time.perf_counter() - search_started

    per_question = []
    ranks: list[int | None] = []
    for question_index in range(n):
        neighbor_ids = [int(chunk_id) for chunk_id in ids[question_index].tolist()]
        if any(chunk_id < 0 or chunk_id >= len(rows) for chunk_id in neighbor_ids):
            raise RuntimeError(f"search returned an invalid chunk id for question {question_index}")
        retrieved_pubids = [_json_pubid(rows[chunk_id]["pubid"]) for chunk_id in neighbor_ids]
        gold_pubid = questions[question_index]["pubid"]
        rank = first_hit_rank(retrieved_pubids, gold_pubid)
        ranks.append(rank)
        per_question.append(
            {
                "question_index": question_index,
                "pubid": _json_pubid(gold_pubid),
                "question": questions[question_index]["question"],
                "first_hit_rank": rank,
                "hit_at_5": rank is not None and rank <= 5,
                "hit_at_10": rank is not None and rank <= 10,
                "reciprocal_rank_at_5": reciprocal_rank(rank, 5),
                "reciprocal_rank_at_10": reciprocal_rank(rank, 10),
                "retrieved_pubids": retrieved_pubids,
            }
        )

    by_k = {k: summarize_ranks(ranks, k) for k in KS}
    misses = [
        {
            "question_index": row["question_index"],
            "pubid": row["pubid"],
            "question": row["question"],
            "retrieved_pubids": row["retrieved_pubids"],
        }
        for row in per_question
        if not row["hit_at_10"]
    ]
    summary = {
        "artifact_dir": str(Path(artifact_dir)),
        "dataset": {
            "id": manifest["dataset"]["id"],
            "config": manifest["dataset"]["config"],
            "split": split_name,
            "requested_revision": dataset_revision,
        },
        "model": {
            "id": manifest["model"]["id"],
            "requested_revision": model_revision,
        },
        "n": n,
        "order": "first train rows, dataset order, no shuffle",
        "batch_size": batch_size,
        "pooling": manifest["pooling"],
        "normalization": manifest["normalization"],
        "ranking": "chunk",
        "deduplicate_pubid": False,
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(device),
            "parameter_device": str(parameter.device),
            "parameter_dtype": str(parameter.dtype),
        },
        "encode_seconds": encode_seconds,
        "search_seconds": search_seconds,
        "metrics": {str(k): by_k[k] for k in KS},
        "misses_at_10": misses,
        "notebook_stdout": {
            "recall_at_5": NOTEBOOK_STDOUT[5]["recall"],
            "mrr_at_5": NOTEBOOK_STDOUT[5]["mrr"],
            "recall_at_10": NOTEBOOK_STDOUT[10]["recall"],
            "mrr_at_10": NOTEBOOK_STDOUT[10]["mrr"],
            "note": (
                "MRR 0.968 is eval_retrieval at k=5. "
                "The k=10 call printed MRR 0.969. Same first-hit definition, different cutoff."
            ),
        },
    }
    written = write_results(out_dir, per_question, summary)
    _print_report(summary, written)
    return written


def _print_report(summary: dict, written: dict[str, Path]) -> None:
    print(f"device {summary['device']['name']} {summary['device']['parameter_device']} {summary['device']['parameter_dtype']}")
    print(f"encode_seconds {summary['encode_seconds']:.3f}")
    print(f"search_seconds {summary['search_seconds']:.3f}")
    for k in KS:
        metrics = summary["metrics"][str(k)]
        notebook = NOTEBOOK_STDOUT[k]
        print(
            f"Recall@{k} = {metrics['recall']:.6f} ({metrics['hits']}/{metrics['n']})"
            f"  notebook {notebook['recall']:.3f}"
        )
        print(
            f"MRR@{k} = {metrics['mrr']:.6f}"
            f"  notebook k={k} {notebook['mrr']:.3f}"
        )
    misses = summary["misses_at_10"]
    print(f"miss_at_10 {len(misses)}")
    for miss in misses:
        print(f"MISS {miss['question_index']}\t{miss['pubid']}\t{miss['question']}")
    for path in written.values():
        print(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PubMedQA IndexFlatIP artifact. Does not rebuild it or call Gemini."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Directory with index.faiss, metadata.jsonl, and manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="New directory for per_question.jsonl and summary.json",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="First n train questions, dataset order (default 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Query encoder batch size (default 8)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluate(
            args.artifact_dir,
            args.out_dir,
            n=args.n,
            batch_size=args.batch_size,
        )
    except FileExistsError as exc:
        print(exc, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
