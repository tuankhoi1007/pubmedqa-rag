"""Pure helpers for the PubMedQA RAG pipeline (no network, no torch)."""

from __future__ import annotations

import re
import string
from collections.abc import Hashable, Sequence
from typing import Any


def clean_and_tokenize(text: str, stop_words: set[str] | None = None) -> list[str]:
    """Lowercase, strip punctuation, split, optionally drop stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if stop_words:
        tokens = [w for w in tokens if w not in stop_words]
    return tokens


def build_context(samples: Any) -> str:
    """Format passages as ``[i] (pubid …) text``.

    Accepts either:
    - a dict with parallel lists ``text`` and ``pubid`` (HF Dataset style), or
    - a sequence of ``(pubid, text)`` pairs.
    """
    if isinstance(samples, dict):
        texts = samples["text"]
        pubids = samples["pubid"]
        pairs = list(zip(pubids, texts))
    else:
        pairs = list(samples)
    lines = [f"[{i}] (pubid {p}) {t}" for i, (p, t) in enumerate(pairs, 1)]
    return "\n\n".join(lines)


_ANSWER_RE = re.compile(r"Answer:\s*(yes|no|maybe)", re.IGNORECASE)


def parse_label(raw: str) -> str:
    """Extract yes/no/maybe from model output; else ``parse_fail``."""
    if not raw:
        return "parse_fail"
    m = _ANSWER_RE.search(raw)
    return m.group(1).lower() if m else "parse_fail"


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 5,
    c: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Hashable]:
    """Reciprocal Rank Fusion; return top-``k`` keys (ranks start at 1)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists length")
    score: dict[Hashable, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, key in enumerate(lst, start=1):
            score[key] = score.get(key, 0.0) + float(w) / (c + rank)
    return sorted(score, key=score.get, reverse=True)[:k]
