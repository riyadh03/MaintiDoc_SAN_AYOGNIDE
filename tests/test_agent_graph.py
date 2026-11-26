"""Graphe LangGraph MaintiDoc : chaque chemin et chaque borne de boucle, sans Ollama ni modèles.

Faux déterministes pour le routeur, le générateur, la recherche et le reranker ;
EvidencePacker et validate_citations sont les VRAIS composants.
"""
from __future__ import annotations

from typing import List

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from maintidoc.agent import MaintiDocAgent, RouteDecision
from maintidoc.agent.graph import sqlite_checkpointer
from maintidoc.config import AGENT_MAX_CLARIFICATIONS
from maintidoc.retrieval import EvidencePacker
from maintidoc.schemas import SearchHit, make_citation_id

HIT = SearchHit(
    element_id="p05_table_01", page=5, modality="table", score=0.9, rank=1,
    metadata={"document_id": "SAN-MAN-001", "content": "| AL-01 | Orange | Chemin FC degrade |"},
)
CITATION = make_citation_id("SAN-MAN-001", 5, "p05_table_01")


def route(intent="documentation", **kw):
    return RouteDecision(intent=intent, reason="test", search_query=kw.pop("search_query", "voyant orange"), **kw)


def answer(citations=(CITATION,), status="answered", **kw):
    return {"status": status, "answer": "AL-01 : chemin FC dégradé.", "citations": list(citations),
            "uncertainties": [], "clarification_question": kw.get("clarification_question")}


class FakeRouter:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.questions: List[str] = []

    def classify(self, question):
        self.questions.append(question)
        d = self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        if isinstance(d, Exception):
            raise d
        return d


class FakeGenerator:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.feedbacks: List = []

    def generate(self, question, texts, images, feedback=None):
        self.feedbacks.append(feedback)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


class FakeRetriever:
    def __init__(self, hits=(HIT,)):
        self.hits = list(hits)
        self.calls: List[dict] = []

    def retrieve(self, question, visual_query_en=None, **kw):
        self.calls.append({"question": question, **kw})
        return list(self.hits), []


class FakeReranker:
    """``results`` : une liste de résultats par appel (la dernière est répétée)."""

    def __init__(self, *results):
        self.results = list(results) or [None]

    def rerank(self, question, hits, top_k=4, min_score=None):
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return hits[:top_k] if r is None else r


def make_agent(router, generator, retriever=None, reranker=None, checkpointer=None):
    return MaintiDocAgent(
        retriever or FakeRetriever(), reranker or FakeReranker(), EvidencePacker(),
        router, generator, checkpointer=checkpointer or InMemorySaver(),
    )


# ─────────────────────────────────────────────────────────────────────────────

def test_documentation_happy_path_returns_cited_answer():
    turn = make_agent(FakeRouter(route()), FakeGenerator(answer())).ask("Que signifie AL-01 ?")

    assert turn.status == "answered"
    assert turn.citations == [CITATION]
    assert turn.sources[0]["page"] == 5 and turn.sources[0]["element_id"] == "p05_table_01"
    assert turn.trace == ["classify:documentation", "retrieve:1", "rerank:1", "pack:1t+0i",
                          "generate:1", "validate:ok", "finalize"]


def test_ambiguous_question_pauses_then_resumes_with_user_answer():
    router = FakeRouter(route("ambiguous", clarification_question="Quel voyant ?"), route())
    agent = make_agent(router, FakeGenerator(answer()))

    first = agent.ask("Ça clignote, que faire ?")
    assert first.status == "needs_clarification"
    assert first.clarification_question == "Quel voyant ?"

    final = agent.reply(first.thread_id, "Le voyant orange, alarme AL-01")
    assert final.status == "answered"
    # La 2e classification voit la précision apportée par l'utilisateur
    assert "Le voyant orange, alarme AL-01" in router.questions[1]


def test_clarification_loop_is_bounded():
    agent = make_agent(FakeRouter(route("ambiguous")), FakeGenerator(answer()))
    turn = agent.ask("Problème ?")
    for i in range(AGENT_MAX_CLARIFICATIONS):
        assert turn.status == "needs_clarification", f"tour {i}"
        turn = agent.reply(turn.thread_id, "je ne sais pas")
    # Plafond atteint : le graphe cherche avec ce qu'il a au lieu de redemander
    assert turn.status == "answered"


def test_hallucinated_citation_is_corrected_on_retry():
    gen = FakeGenerator(answer(citations=["SAN-MAN-001:p9:invente"]), answer())
    turn = make_agent(FakeRouter(route()), gen).ask("Que signifie AL-01 ?")

    assert turn.status == "answered"
    assert gen.feedbacks[0] is None
    assert "invente" in gen.feedbacks[1]  # le validateur explique au LLM ce qui est rejeté
    assert "validate:ko" in turn.trace and turn.trace.count("generate:1") == 1


def test_persistent_hallucination_is_rejected():
    turn = make_agent(FakeRouter(route()), FakeGenerator(answer(citations=["faux"]))).ask("AL-01 ?")

    assert turn.status == "insufficient_evidence"
    assert turn.trace[-1] == "reject"
    assert any("faux" in u for u in turn.uncertainties)


def test_generation_failure_is_retried_then_rejected():
    turn = make_agent(FakeRouter(route()), FakeGenerator(RuntimeError("Ollama down"))).ask("AL-01 ?")
    assert turn.status == "insufficient_evidence"
    assert any("Ollama down" in u for u in turn.uncertainties)


def test_no_evidence_rewrites_query_once_then_gives_up():
    retriever = FakeRetriever()
    turn = make_agent(FakeRouter(route(search_query="mots cles")), FakeGenerator(answer()),
                      retriever=retriever, reranker=FakeReranker([])).ask("Question introuvable")

    assert turn.status == "insufficient_evidence"
    assert len(retriever.calls) == 2
    assert retriever.calls[0]["question"] == "mots cles"
    assert retriever.calls[1]["question"] == "Question introuvable"  # repart de la question complète
    assert retriever.calls[1]["fused_top_n"] > retriever.calls[0]["fused_top_n"]  # fenêtre élargie


def test_rewrite_recovers_evidence():
    turn = make_agent(FakeRouter(route()), FakeGenerator(answer()),
                      reranker=FakeReranker([], None)).ask("AL-01 ?")
    assert turn.status == "answered"
    assert "rewrite_query" in turn.trace


@pytest.mark.parametrize("intent, status", [
    ("out_of_scope", "out_of_scope"),
    ("operational_data", "tool_unavailable"),
    ("action_request", "action_refused"),
])
def test_non_documentary_intents_skip_retrieval(intent, status):
    retriever = FakeRetriever()
    turn = make_agent(FakeRouter(route(intent)), FakeGenerator(answer()), retriever=retriever).ask("x")
    assert turn.status == status
    assert retriever.calls == []


def test_router_failure_falls_back_to_documentation():
    turn = make_agent(FakeRouter(ConnectionError("ollama")), FakeGenerator(answer())).ask("AL-01 ?")
    assert turn.status == "answered" and turn.intent == "documentation"


def test_llm_can_request_clarification():
    gen = FakeGenerator(answer(citations=[], status="needs_clarification", clarification_question="Quel contrôleur ?"),
                        answer())
    agent = make_agent(FakeRouter(route()), gen)
    turn = agent.ask("Le contrôleur est en défaut")
    assert turn.status == "needs_clarification"
    assert turn.clarification_question == "Quel contrôleur ?"
    assert agent.reply(turn.thread_id, "Contrôleur B").status == "answered"


def test_reply_without_pending_clarification_raises():
    agent = make_agent(FakeRouter(route()), FakeGenerator(answer()))
    turn = agent.ask("AL-01 ?")
    with pytest.raises(ValueError):
        agent.reply(turn.thread_id, "réponse")


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        make_agent(FakeRouter(route()), FakeGenerator(answer())).ask("   ")


def test_clarification_citing_manual_code_stops_the_loop():
    """Bug vu en live (Qwen 3B) : le routeur jugeait encore « ambigu » une précision citant AL-01."""
    router = FakeRouter(route("ambiguous", clarification_question="Quel voyant ?"))  # toujours « ambigu »
    agent = make_agent(router, FakeGenerator(answer()))
    first = agent.ask("Ça clignote en orange, que faire ?")
    final = agent.reply(first.thread_id, "Voyant orange, alarme AL-01 sur le fabric A")

    assert final.status == "answered"
    assert router.questions[1].endswith("Précisions de l'utilisateur : Voyant orange, alarme AL-01 sur le fabric A")


def test_cited_answer_flagged_insufficient_is_answered_with_uncertainty():
    """Bug vu en live : réponse + citations valides mais statut 'insufficient_evidence'."""
    gen = FakeGenerator(answer(status="insufficient_evidence"))
    turn = make_agent(FakeRouter(route()), gen).ask("Que signifie AL-01 ?")
    assert turn.status == "answered"
    assert turn.citations == [CITATION]
    assert any("partielles" in u for u in turn.uncertainties)


def test_pending_clarification_survives_restart(tmp_path):
    """Checkpoint SQLite : un nouveau processus reprend la conversation en attente."""
    db = tmp_path / "checkpoints.sqlite"
    router = FakeRouter(route("ambiguous", clarification_question="Quel voyant ?"), route())

    first = make_agent(router, FakeGenerator(answer()), checkpointer=sqlite_checkpointer(db)).ask("Ça clignote")
    assert first.status == "needs_clarification"

    restarted = make_agent(router, FakeGenerator(answer()), checkpointer=sqlite_checkpointer(db))
    assert restarted.reply(first.thread_id, "Orange").status == "answered"
