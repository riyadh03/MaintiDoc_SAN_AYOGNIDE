"""Tests du CrossEncoderReranker avec un faux modèle injecté (aucun téléchargement)."""
from maintidoc.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Simule un cross-encoder : score = fraction des mots de la question présents dans le passage."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        out = []
        for question, passage in pairs:
            q = set(question.lower().split())
            out.append(len(q & set(passage.lower().split())) / len(q))
        return out


def _hits(make_hit):
    return [
        make_hit("A", 1, content="procédure de remplacement du disque"),
        make_hit("B", 2, content="voyant orange alimentation"),
        make_hit("C", 3, content="voyant orange contrôleur alimentation défaut"),
    ]


def test_rerank_reorders_and_traces(make_hit):
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    top = reranker.rerank("voyant orange alimentation défaut", _hits(make_hit), top_k=2)

    assert [h.element_id for h in top] == ["C", "B"]
    assert [h.rank for h in top] == [1, 2]
    assert top[0].metadata["pre_rerank_rank"] == 3
    assert top[0].metadata["reranker_score"] == top[0].score == 1.0


def test_min_score_filters_irrelevant_passages(make_hit):
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    top = reranker.rerank("voyant orange alimentation défaut", _hits(make_hit), top_k=10, min_score=0.5)

    assert "A" not in [h.element_id for h in top]


def test_nothing_above_threshold_returns_empty(make_hit):
    """Liste vide = signal 'preuves insuffisantes' pour l'orchestrateur."""
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    assert reranker.rerank("firmware zoning fabric", _hits(make_hit), min_score=0.1) == []


def test_empty_hits_does_not_call_model():
    fake = FakeCrossEncoder()
    assert CrossEncoderReranker(model=fake).rerank("q", []) == []
    assert fake.calls == 0


def test_default_model_is_multilingual():
    assert CrossEncoderReranker()._model_name == "BAAI/bge-reranker-v2-m3"
