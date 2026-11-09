import numpy as np
import pytest


def test_embed_texts_uses_cls_l2_and_the_raw_string():
    import torch

    from rag.embedding import embed_texts

    seen = {}

    class Tokenizer:
        def __call__(self, texts, padding, truncation, return_tensors):
            seen["texts"] = list(texts)
            seen["padding"] = padding
            seen["truncation"] = truncation
            seen["return_tensors"] = return_tensors
            return {"input_ids": torch.ones(len(texts), 2)}

    class Model(torch.nn.Module):
        def forward(self, input_ids):
            hidden = torch.tensor(
                [[[3.0, 4.0, 0.0], [100.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
            return type("Out", (), {"last_hidden_state": hidden})()

    model = Model()
    model.train()
    vectors = embed_texts(
        ["lace plant"],
        Tokenizer(),
        model,
        torch.device("cpu"),
        batch_size=8,
    )

    assert seen == {
        "texts": ["lace plant"],
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    assert vectors.shape == (1, 3)
    assert vectors.dtype == np.float32
    assert vectors[0].tolist() == pytest.approx([0.6, 0.8, 0.0])
    assert model.training is False
