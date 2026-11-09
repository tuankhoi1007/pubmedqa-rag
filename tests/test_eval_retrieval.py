import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_retrieval


def test_eval_uses_the_shared_encoder():
    from rag.embedding import embed_texts

    assert eval_retrieval.embed_texts is embed_texts


def test_first_hit_keeps_chunk_rank_when_pubids_repeat():
    assert eval_retrieval.first_hit_rank([10, 10, 20], 10) == 1
    assert eval_retrieval.first_hit_rank([20, 20, 10], 10) == 3
    assert eval_retrieval.first_hit_rank([7, 7, 7, 9], 9) == 4
    assert eval_retrieval.first_hit_rank([1, 2, 3], 9) is None


def test_cutoff_uses_first_hit_and_zero_when_missing():
    ranks = [1, 4, 6, None]
    at_5 = eval_retrieval.summarize_ranks(ranks, 5)
    at_10 = eval_retrieval.summarize_ranks(ranks, 10)

    assert at_5["hits"] == 2
    assert at_5["recall"] == pytest.approx(2 / 4)
    assert at_5["mrr"] == pytest.approx((1.0 + 1.0 / 4 + 0.0 + 0.0) / 4)
    assert at_10["hits"] == 3
    assert at_10["recall"] == pytest.approx(3 / 4)
    assert at_10["mrr"] == pytest.approx((1.0 + 0.25 + 1.0 / 6 + 0.0) / 4)
    assert eval_retrieval.reciprocal_rank(5, 5) == pytest.approx(0.2)
    assert eval_retrieval.reciprocal_rank(6, 5) == 0.0
    assert eval_retrieval.reciprocal_rank(None, 10) == 0.0


def test_duplicate_pubids_are_not_collapsed_before_scoring():
    retrieved = [4, 4, 4, 9]
    rank = eval_retrieval.first_hit_rank(retrieved, 9)
    assert rank == 4
    assert eval_retrieval.reciprocal_rank(rank, 5) == pytest.approx(1 / 4)
    collapsed = list(dict.fromkeys(retrieved))
    assert eval_retrieval.first_hit_rank(collapsed, 9) == 2


def test_existing_result_files_are_not_overwritten(tmp_path):
    out_dir = tmp_path / "results"
    written = eval_retrieval.write_results(
        out_dir,
        [{"question_index": 0, "pubid": 1}],
        {"n": 1},
    )
    before = written["per_question"].read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        eval_retrieval.write_results(
            out_dir,
            [{"question_index": 0, "pubid": 2}],
            {"n": 2},
        )

    assert written["per_question"].read_text(encoding="utf-8") == before
    assert '"pubid": 1' in before
    assert '"n": 1' in written["summary"].read_text(encoding="utf-8")
