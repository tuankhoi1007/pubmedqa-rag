from rag.core import build_context


def test_build_context_from_dict():
    samples = {"text": ["hello", "world"], "pubid": [111, 222]}
    ctx = build_context(samples)
    assert "[1]" in ctx and "[2]" in ctx
    assert "pubid 111" in ctx and "pubid 222" in ctx
    assert "hello" in ctx and "world" in ctx


def test_build_context_from_pairs():
    ctx = build_context([(1, "a"), (2, "b")])
    assert ctx.startswith("[1] (pubid 1) a")
