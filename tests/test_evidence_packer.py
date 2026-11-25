"""Tests de l'EvidencePacker (budget, citations, images)."""
import pytest

from maintidoc.retrieval.evidence_packer import EvidencePacker
from maintidoc.schemas import SearchHit


def _image_hit(element_id="p02_image_01", page=2, image_path="data/processed/images/p02_image_01.png"):
    return SearchHit(
        element_id=element_id,
        page=page,
        modality="schema",
        score=0.3,
        rank=1,
        metadata={"document_id": "SAN-MAN-001", "figure_id": "FIG-01", "image_path": image_path},
    )


def test_skips_oversized_passage_and_continues(make_hit):
    """Exercice du cours : 3000 / 4000 / 500 tokens avec un budget de 6000 → on garde 1 et 3."""
    hits = [
        make_hit("A", 1, n_chars=3000 * 4),
        make_hit("B", 2, n_chars=4000 * 4),
        make_hit("C", 3, n_chars=500 * 4),
    ]

    texts, _, _ = EvidencePacker(max_tokens=6000, max_images=0).pack(hits, [])

    assert [t["element_id"] for t in texts] == ["A", "C"]


def test_citation_ids_are_canonical(make_hit):
    texts, _, citations = EvidencePacker().pack([make_hit("p03_text_01", 1, page=3)], [])

    assert texts[0]["citation_id"] == "SAN-MAN-001:p3:p03_text_01"
    assert list(citations) == ["SAN-MAN-001:p3:p03_text_01"]
    assert citations["SAN-MAN-001:p3:p03_text_01"]["page"] == 3


def test_two_elements_on_same_page_get_distinct_citations(make_hit):
    hits = [make_hit("p03_text_01", 1, page=3), make_hit("p03_table_01", 2, page=3, modality="table")]

    _, _, citations = EvidencePacker().pack(hits, [])

    assert len(citations) == 2


def test_image_budget_is_reserved_before_texts(make_hit):
    """Un long texte ne peut pas évincer le schéma : le budget image est réservé d'abord."""
    hits = [make_hit("A", 1, n_chars=1000 * 4)]

    texts, images, citations = EvidencePacker(max_tokens=1000, max_images=1).pack(hits, [_image_hit()])

    assert texts == []  # 1000 tokens de texte > 1000 - 500 réservés à l'image
    assert len(images) == 1
    assert images[0]["citation_id"] == "SAN-MAN-001:p2:p02_image_01"
    assert images[0]["figure_id"] == "FIG-01"
    assert "SAN-MAN-001:p2:p02_image_01" in citations


def test_images_without_path_are_ignored():
    _, images, citations = EvidencePacker().pack([], [_image_hit(image_path="")])
    assert images == [] and citations == {}


def test_empty_content_and_duplicates_are_skipped(make_hit):
    hits = [make_hit("A", 1, content="   "), make_hit("B", 2), make_hit("B", 3)]

    texts, _, _ = EvidencePacker().pack(hits, [])

    assert [t["element_id"] for t in texts] == ["B"]


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_images": -1}, {"max_tokens": 400, "max_images": 1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EvidencePacker(**kwargs)
