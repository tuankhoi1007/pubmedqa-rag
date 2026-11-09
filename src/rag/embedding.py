"""Shared dense encoder for the PubMedQA index.

Build, retrieval evaluation, and ``RAGEngine`` all call ``embed_texts``.
The notebook settings are CLS pooling, L2 normalization, and no query prefix.
``max_length`` is not passed. The forward pass runs under ``model.eval()``
and ``torch.inference_mode()``.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "BAAI/bge-small-en-v1.5"
POOLING = "cls"
NORMALIZATION = "l2"
METRIC = "inner_product"
INDEX_TYPE = "IndexFlatIP"
DTYPE = "float32"
QUERY_BATCH_SIZE = 8


def embed_texts(texts, tokenizer, model, device, batch_size: int) -> np.ndarray:
    """Encode texts with CLS pooling and L2 normalization.

    No query prefix is added. ``max_length`` is not passed, matching the
    notebook call. ``model.eval()`` and ``torch.inference_mode()`` wrap the
    forward pass.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not texts:
        raise ValueError("no texts to embed")

    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            cls = model(**encoded).last_hidden_state[:, 0]
            normed = torch.nn.functional.normalize(cls, p=2, dim=-1)
            parts.append(np.ascontiguousarray(normed.cpu().numpy(), dtype=np.float32))
    return np.concatenate(parts, axis=0)


class TextEmbedder:
    """Call ``embed_texts`` with one tokenizer, model, and device."""

    def __init__(self, tokenizer, model, device, batch_size: int = QUERY_BATCH_SIZE) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return embed_texts(texts, self.tokenizer, self.model, self.device, self.batch_size)
