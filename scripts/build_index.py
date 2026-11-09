"""Build the PubMedQA dense index from notebooks/Thực_hành_clean.ipynb.

Settings taken from that notebook, not invented:

- dataset ``load_dataset("qiaojin/PubMedQA", "pqa_labeled")``, split ``train``
- one chunk per ``example["context"]["contexts"]`` item, train order then
  context order
- ``AutoTokenizer`` + ``AutoModel``, ``BAAI/bge-small-en-v1.5``
- CLS pooling, L2 normalization, no query prefix
- tokenizer call is ``padding=True, truncation=True`` with no ``max_length``

The encoder implementation is ``rag.embedding.embed_texts``.

The notebook did not pin a dataset or model revision and did not name a
finite max length. Each Hub SHA is resolved once and passed into that load.
``manifest.json`` records the requested revision and, when the loaded object
exposes one, the observed commit. A dataset fingerprint is not a commit.
This script does not call Gemini.

Not declared in pyproject.toml (needed for a real build): numpy, faiss-cpu,
torch, transformers, datasets, huggingface_hub.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path

import numpy as np

from rag.embedding import (
    DTYPE,
    INDEX_TYPE,
    METRIC,
    MODEL_ID,
    NORMALIZATION,
    POOLING,
    embed_texts,
)

DATASET_ID = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"
DATASET_SPLIT = "train"

ARTIFACTS = ("index.faiss", "metadata.jsonl", "manifest.json")


def iter_chunks(examples):
    """Yield ``(pubid, text)`` in train order, then context order."""
    for example in examples:
        pubid = example["pubid"]
        for text in example["context"]["contexts"]:
            yield pubid, text


def describe_truncation(tokenizer) -> dict:
    """Record the notebook truncation call and the max length it will use.

    Transformers, when ``truncation=True`` and ``max_length`` is omitted, uses
    ``tokenizer.model_max_length`` unless that value is above ``LARGE_INTEGER``,
    in which case it does not truncate. The threshold is imported from the
    installed library. A finite length is not hardcoded.
    """
    from transformers.tokenization_utils_base import LARGE_INTEGER

    raw = getattr(tokenizer, "model_max_length", None)
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        raise RuntimeError(
            "tokenizer.model_max_length is not an integer; refusing to guess a max length"
        )
    raw = int(raw)
    if raw > int(LARGE_INTEGER):
        effective = None
    else:
        effective = raw
    return {
        "padding": True,
        "truncation": True,
        "max_length_argument": None,
        "model_max_length": raw,
        "effective_max_length": effective,
    }


def resolve_hub_sha(repo_id: str, repo_type: str) -> str:
    """Resolve one Hub commit SHA. Call once per repository."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha.strip():
        raise RuntimeError(f"{repo_type} {repo_id} did not return a commit SHA")
    return sha.strip()


def optional_commit_hash(obj) -> str | None:
    """Return ``_commit_hash`` when that attribute exists. Absence is allowed."""
    if obj is None:
        return None
    value = getattr(obj, "_commit_hash", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def dataset_observed_commit(dataset) -> str | None:
    """Read a git commit from the loaded dataset, never its fingerprint."""
    for owner in (dataset, getattr(dataset, "info", None)):
        if owner is None:
            continue
        for name in ("commit_hash", "observed_commit_hash"):
            if "fingerprint" in name:
                continue
            value = getattr(owner, name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def accept_observed_commit(kind: str, requested: str, observed: str | None) -> str | None:
    """Keep an observed commit only when it matches the requested SHA."""
    if observed is None:
        return None
    if observed != requested:
        raise RuntimeError(
            f"{kind} observed commit {observed} != requested revision {requested}"
        )
    return observed


def load_train_split():
    """Load ``pqa_labeled`` train at the dataset SHA resolved for this call."""
    from datasets import load_dataset

    dataset_revision = resolve_hub_sha(DATASET_ID, "dataset")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=dataset_revision)
    if DATASET_SPLIT not in dataset:
        raise KeyError(f"split {DATASET_SPLIT!r} is not in the loaded dataset")
    observed = accept_observed_commit(
        "dataset",
        dataset_revision,
        dataset_observed_commit(dataset[DATASET_SPLIT]),
    )
    return dataset[DATASET_SPLIT], dataset_revision, observed


def load_encoder():
    """Load the notebook embedder at the model SHA resolved for this call."""
    from transformers import AutoModel, AutoTokenizer

    model_revision = resolve_hub_sha(MODEL_ID, "model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
    model = AutoModel.from_pretrained(MODEL_ID, revision=model_revision)
    observed_values = [
        commit
        for commit in (
            optional_commit_hash(tokenizer),
            optional_commit_hash(getattr(model, "config", None)),
        )
        if commit is not None
    ]
    if len(set(observed_values)) > 1:
        raise RuntimeError(
            "tokenizer and model observed commits differ: " + ", ".join(observed_values)
        )
    observed = accept_observed_commit(
        "model",
        model_revision,
        observed_values[0] if observed_values else None,
    )
    model.eval()
    return tokenizer, model, model_revision, observed, describe_truncation(tokenizer)


def library_versions() -> dict[str, str]:
    import faiss

    versions = {
        "numpy": version("numpy"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "huggingface_hub": version("huggingface_hub"),
    }
    faiss_version = getattr(faiss, "__version__", None)
    for dist in ("faiss-cpu", "faiss-gpu"):
        try:
            faiss_version = version(dist)
            break
        except PackageNotFoundError:
            continue
    if not faiss_version:
        raise RuntimeError("faiss is imported but its version could not be read")
    versions["faiss"] = faiss_version
    return versions


def _revision_record(identity: dict, requested: str, observed: str | None) -> dict:
    record = dict(identity)
    record["requested_revision"] = requested
    if observed is not None:
        record["observed_commit_hash"] = observed
    return record


def assemble_manifest(
    *,
    dataset_revision: str,
    model_revision: str,
    dimension: int,
    num_chunks: int,
    truncation: dict,
    library_versions: dict[str, str],
    dataset_observed_commit: str | None = None,
    model_observed_commit: str | None = None,
) -> dict:
    if not dataset_revision or not model_revision:
        raise ValueError("dataset_revision and model_revision are required")
    if dimension < 1 or num_chunks < 1:
        raise ValueError("dimension and num_chunks must be positive")
    return {
        "dataset": _revision_record(
            {
                "id": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
            },
            dataset_revision,
            dataset_observed_commit,
        ),
        "model": _revision_record(
            {"id": MODEL_ID},
            model_revision,
            model_observed_commit,
        ),
        "pooling": POOLING,
        "normalization": NORMALIZATION,
        "metric": METRIC,
        "index": INDEX_TYPE,
        "dtype": DTYPE,
        "dimension": dimension,
        "num_chunks": num_chunks,
        "truncation": truncation,
        "library_versions": library_versions,
    }


def _metadata_row(chunk_id: int, pubid, text) -> dict:
    if not isinstance(text, str):
        raise TypeError("chunk text must be a str")
    if isinstance(pubid, bool) or not isinstance(pubid, (str, Integral)):
        raise TypeError("pubid must be a string or an integer")
    if not isinstance(pubid, str):
        pubid = int(pubid)
    return {"chunk_id": chunk_id, "pubid": pubid, "text": text}


def make_flat_ip(vectors: np.ndarray):
    import faiss

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must have shape (n, d) with n and d > 0")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {name: out_dir / name for name in ARTIFACTS}


def save_artifacts(out_dir, vectors, records, manifest, *, force: bool = False) -> dict[str, Path]:
    """Write index.faiss, metadata.jsonl, and manifest.json.

    ``chunk_id`` is the row added to the index. Existing artifact files are
    left untouched unless ``force`` is true.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(out_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"artifacts already exist (pass --force to overwrite): {listed}")

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if len(records) != matrix.shape[0]:
        raise ValueError("records and vectors must have the same length")
    if manifest["dimension"] != matrix.shape[1] or manifest["num_chunks"] != len(records):
        raise ValueError("manifest dimension or num_chunks does not match the index")

    rows = [
        _metadata_row(chunk_id, record["pubid"], record["text"])
        for chunk_id, record in enumerate(records)
    ]
    index = make_flat_ip(matrix)
    if index.ntotal != len(rows):
        raise RuntimeError("index size does not match metadata")

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        metadata_path = tmp_dir / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        import faiss

        faiss.write_index(index, str(tmp_dir / "index.faiss"))
        for name, dest in paths.items():
            os.replace(tmp_dir / name, dest)
    return paths


def load_index(path):
    import faiss

    return faiss.read_index(str(path))


def read_metadata(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def search_index(index, query: np.ndarray, k: int):
    if k < 1:
        raise ValueError("k must be positive")
    matrix = np.ascontiguousarray(query, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return index.search(matrix, k)


def build(out_dir, *, force: bool = False, batch_size: int = 32) -> dict[str, Path]:
    """Load the corpus and embedder, then write the three artifacts.

    ``batch_size`` is only a memory knob. The notebook did not set one; CLS
    pooling does not depend on it.
    """
    import torch

    train, dataset_revision, dataset_observed_commit = load_train_split()
    records = [{"pubid": pubid, "text": text} for pubid, text in iter_chunks(train)]
    if not records:
        raise RuntimeError("train split produced no chunks")

    tokenizer, model, model_revision, model_observed_commit, truncation = load_encoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    vectors = embed_texts(
        [record["text"] for record in records],
        tokenizer,
        model,
        device,
        batch_size,
    )
    manifest = assemble_manifest(
        dataset_revision=dataset_revision,
        model_revision=model_revision,
        dataset_observed_commit=dataset_observed_commit,
        model_observed_commit=model_observed_commit,
        dimension=int(vectors.shape[1]),
        num_chunks=len(records),
        truncation=truncation,
        library_versions=library_versions(),
    )
    return save_artifacts(out_dir, vectors, records, manifest, force=force)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PubMedQA IndexFlatIP index. Does not call Gemini."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for index.faiss, metadata.jsonl, and manifest.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite index.faiss, metadata.jsonl, and manifest.json if they exist",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoder batch size (not set by the notebook; default 32)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        written = build(args.out_dir, force=args.force, batch_size=args.batch_size)
    except FileExistsError as exc:
        print(exc, flush=True)
        return 2
    for path in written.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
