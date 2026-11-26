"""Graphe + MCP + human-in-the-loop, de bout en bout (sans Ollama) :
faux routeur, VRAI client MCP, VRAI serveur MCP, VRAIE GMAO SQLite en mémoire.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from maintidoc.agent.graph import sqlite_checkpointer
from maintidoc.gmao import GMAOStore, MCPGMAOClient
from maintidoc.gmao.server import build_server
from test_agent_graph import FakeGenerator, FakeRouter, FakeReranker, FakeRetriever, answer, route

from maintidoc.agent import MaintiDocAgent
from maintidoc.retrieval import EvidencePacker


@pytest.fixture
def store():
    return GMAOStore()


def make_agent(store, intent, checkpointer=None, gmao="mcp"):
    gmao_client = MCPGMAOClient(build_server(store)) if gmao == "mcp" else gmao
    return MaintiDocAgent(FakeRetriever(), FakeReranker(), EvidencePacker(), FakeRouter(route(intent)),
                          FakeGenerator(answer()), checkpointer=checkpointer or InMemorySaver(), gmao=gmao_client)


# ── Lecture (pas de validation humaine) ─────────────────────────────────────

def test_open_tickets_question_is_answered_from_gmao(store):
    turn = make_agent(store, "operational_data").ask("Quels tickets sont ouverts sur le contrôleur B ?")
    assert turn.status == "answered"
    assert turn.citations == ["GMAO:INC-130"]
    assert "INC-130" in turn.answer
    assert turn.trace == ["classify:operational_data", "tool:get_open_tickets"]


def test_history_question_uses_history_tool(store):
    turn = make_agent(store, "operational_data").ask("Le disque B03 a-t-il un antécédent ?")
    assert turn.trace[-1] == "tool:get_equipment_history"
    assert turn.citations == ["GMAO:INC-117"]


def test_gmao_down_gives_tool_error_not_a_crash(store):
    class Down:
        def __getattr__(self, name):
            def fail(*a, **k):
                from maintidoc.gmao import GMAOToolError
                raise GMAOToolError("serveur MCP injoignable")
            return fail
    turn = make_agent(store, "operational_data", gmao=Down()).ask("Tickets ouverts ?")
    assert turn.status == "tool_error" and "injoignable" in turn.answer


# ── Action engageante : brouillon → humain → création ──────────────────────

def test_ticket_request_pauses_for_approval_and_writes_nothing(store):
    turn = make_agent(store, "action_request").ask("Crée un ticket : alarme AL-01 sur le fabric A")

    assert turn.status == "needs_approval"
    draft = turn.pending_ticket
    assert draft["equipment_id"] == "FABRIC-A"
    assert draft["title"] == "Alerte AL-01 sur Fabric Fibre Channel A"  # format du ticket exemple (manuel p.7)
    assert draft["evidence"] == ["AL-01"]
    # Rien n'est écrit tant que l'humain n'a pas décidé
    assert store.get_open_tickets("FABRIC-A") == []
    assert all(e["tool"] != "create_ticket" for e in store.audit_log())


def test_approval_creates_ticket_with_approver_and_audit(store):
    agent = make_agent(store, "action_request")
    pending = agent.ask("Crée un ticket pour le contrôleur B en défaut")
    assert "INC-130" in pending.pending_ticket["evidence"]  # ticket déjà ouvert : signalé
    assert pending.pending_ticket["warnings"]

    done = agent.approve(pending.thread_id, approver="tech.dupont", edits={"priority": "P2"})

    assert done.status == "ticket_created"
    assert done.ticket["id"] == "INC-131"
    assert done.ticket["approved_by"] == "tech.dupont"
    assert done.ticket["priority"] == "P2"  # correction humaine appliquée
    assert done.trace[-2:] == ["approval:approved", "create_ticket:INC-131"]
    created = [e for e in store.audit_log(pending.thread_id) if e["tool"] == "create_ticket"]
    assert [e["outcome"] for e in created] == ["created: INC-131"]


def test_rejection_creates_nothing(store):
    agent = make_agent(store, "action_request")
    pending = agent.ask("Crée un ticket pour le contrôleur B")
    done = agent.reject(pending.thread_id, approver="tech.dupont", reason="Doublon de INC-130")

    assert done.status == "ticket_rejected"
    assert "Doublon de INC-130" in done.answer
    assert [t["id"] for t in store.get_open_tickets("CTRL-B")] == ["INC-130"]


def test_approval_must_be_named(store):
    agent = make_agent(store, "action_request")
    pending = agent.ask("Crée un ticket pour le contrôleur B")
    with pytest.raises(ValueError, match="nominative"):
        agent.approve(pending.thread_id, approver="  ")


def test_cannot_approve_a_clarification_or_reply_to_an_approval(store):
    agent = make_agent(store, "action_request")
    pending = agent.ask("Crée un ticket pour le contrôleur B")
    with pytest.raises(ValueError):
        agent.reply(pending.thread_id, "oui")  # une approbation n'est pas une simple réponse texte


def test_forbidden_action_is_refused_without_any_tool_call(store):
    turn = make_agent(store, "action_request").ask("Fais un failover vers le contrôleur A")
    assert turn.status == "action_refused"
    assert "section 5" in turn.answer
    assert store.audit_log() == []


def test_replayed_creation_does_not_duplicate(store):
    """Crash après la création, puis rejeu du nœud : la clé d'idempotence évite le doublon."""
    agent = make_agent(store, "action_request")
    pending = agent.ask("Crée un ticket pour le contrôleur B")
    agent.approve(pending.thread_id, approver="tech.dupont")

    # Rejeu du même brouillon approuvé, même conversation (comme le ferait un nœud ré-exécuté)
    from maintidoc.agent import actions
    state = agent.graph.get_state(agent._config(pending.thread_id)).values
    key = actions.idempotency_key(pending.thread_id, state["ticket_draft"])
    ticket, created = store.create_ticket(
        **{k: state["ticket_draft"][k] for k in ("equipment_id", "title", "description", "priority", "evidence")},
        approved_by="tech.dupont", idempotency_key=key, thread_id=pending.thread_id,
    )
    assert created is False and ticket["id"] == "INC-131"
    assert len([t for t in store.get_open_tickets("CTRL-B") if t["id"] == "INC-131"]) == 1


def test_end_to_end_over_stdio_subprocess(tmp_path):
    """Chemin de production : le serveur MCP tourne dans un sous-processus (transport stdio)."""
    gmao = MCPGMAOClient.stdio(tmp_path / "gmao.sqlite")
    agent = make_agent(None, "action_request", gmao=gmao)
    pending = agent.ask("Crée un ticket : alarme AL-01 sur le fabric A")
    assert pending.status == "needs_approval"
    done = agent.approve(pending.thread_id, approver="tech.dupont")
    assert done.status == "ticket_created" and done.ticket["id"] == "INC-131"
    # La création est bien persistée côté serveur, dans SA base
    assert any(t["id"] == "INC-131" for t in GMAOStore(tmp_path / "gmao.sqlite").get_open_tickets("FABRIC-A"))


def test_pending_approval_survives_restart(store, tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    pending = make_agent(store, "action_request", checkpointer=sqlite_checkpointer(db)).ask(
        "Crée un ticket pour le contrôleur B")
    restarted = make_agent(store, "action_request", checkpointer=sqlite_checkpointer(db))
    assert restarted.approve(pending.thread_id, approver="tech.dupont").status == "ticket_created"
