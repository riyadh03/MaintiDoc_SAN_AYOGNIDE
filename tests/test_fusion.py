

"""Tests de la fusion RRF par élément (fuse_hits_rrf)."""
import pytest

from maintidoc.retrieval.hybrid_retriever import fuse_hits_rrf


def test_element_found_by_both_sources_ranks_first(make_hit):
    dense = [make_hit("A", 1), make_hit("B", 2)]
    bm25 = [make_hit("B", 1), make_hit("C", 2)]

    fused = fuse_hits_rrf([("dense", dense), ("bm25", bm25)], k=60)

    # B : 1/62 + 1/61 > A : 1/61 > C : 1/62
    assert [h.element_id for h in fused] == ["B", "A", "C"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    assert [h.rank for h in fused] == [1, 2, 3]


def test_raw_scores_are_ignored(make_hit):
    """Un score BM25 énorme ne doit pas influencer la fusion : seuls les rangs comptent."""
    dense = [make_hit("A", 1, score=0.91)]
    bm25 = [make_hit("B", 1, score=42.0)]

    fused = fuse_hits_rrf([("dense", dense), ("bm25", bm25)])

    assert fused[0].score == fused[1].score  # même rang → même contribution RRF


def test_provenance_is_recorded(make_hit):
    dense = [make_hit("A", 3)]
    bm25 = [make_hit("A", 1, source="bm25")]

    hit = fuse_hits_rrf([("dense", dense), ("bm25", bm25)])[0]

    assert hit.metadata["retrieval_sources"] == ["bm25", "dense"]
    assert hit.metadata["source_ranks"] == {"dense": 3, "bm25": 1}
    assert "source" not in hit.metadata  # l'ancienne étiquette mono-source est retirée


def test_duplicate_in_same_list_counts_once(make_hit):
    """Des points Qdrant dupliqués ne doivent pas gonfler le score RRF."""
    dense = [make_hit("A", 1), make_hit("A", 2), make_hit("B", 3)]

    fused = fuse_hits_rrf([("dense", dense)], k=60)

    assert len(fused) == 2
    assert fused[0].score == pytest.approx(1 / 61, abs=1e-6)
    assert fused[0].metadata["source_ranks"] == {"dense": 1}


def test_top_n_truncates(make_hit):
    dense = [make_hit(f"E{i}", i) for i in range(1, 11)]
    assert len(fuse_hits_rrf([("dense", dense)], top_n=3)) == 3


def test_empty_inputs():
    assert fuse_hits_rrf([("dense", []), ("bm25", [])]) == []
