from rag.core import rrf_fuse


def test_rrf_top_k_and_unique():
    out = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=3)
    assert len(out) == 3
    assert len(set(out)) == 3


def test_rrf_both_lists_win():
    # b is #2 dense and #1 bm25 → should beat a (#1 dense only)
    out = rrf_fuse([["a", "b", "c"], ["b", "d", "e"]], k=2, c=60)
    assert out[0] == "b"
