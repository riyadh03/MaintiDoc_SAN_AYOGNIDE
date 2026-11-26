"""GMAO simulée + serveur MCP : règles métier, idempotence, audit, contrat des outils."""
from __future__ import annotations

import pytest

from maintidoc.agent import actions
from maintidoc.gmao import GMAOError, GMAOStore, GMAOToolError, MCPGMAOClient
from maintidoc.gmao.server import build_server

TICKET = dict(equipment_id="FABRIC-A", title="Alerte AL-01 sur Fabric Fibre Channel A",
              description="Chemin FC dégradé sur HBA1", priority="P2", evidence=["AL-01", "INC-104"])


@pytest.fixture
def store():
    return GMAOStore()


@pytest.fixture
def client(store):
    return MCPGMAOClient(build_server(store))  # vrai protocole MCP, transport en mémoire


# ── Store ───────────────────────────────────────────────────────────────────

def test_seed_matches_manual_history(store):
    assert [t["id"] for t in store.get_equipment_history("FABRIC-A")["tickets"]] == ["INC-104"]
    assert {t["id"] for t in store.get_open_tickets()} == {"INC-117", "INC-125", "INC-130"}  # INC-104 est clos


def test_create_requires_named_approver(store):
    with pytest.raises(GMAOError, match="approbation"):
        store.create_ticket(**TICKET, approved_by="  ", idempotency_key="k")
    assert store.get_open_tickets("FABRIC-A") == []


def test_create_is_idempotent(store):
    t1, created1 = store.create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k1")
    t2, created2 = store.create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k1")
    assert (created1, created2) == (True, False)
    assert t1["id"] == t2["id"] == "INC-131"
    assert len(store.get_open_tickets("FABRIC-A")) == 1


def test_same_key_different_content_is_a_conflict(store):
    store.create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k1")
    with pytest.raises(GMAOError, match="Conflit"):
        store.create_ticket(**{**TICKET, "priority": "P1"}, approved_by="tech.a", idempotency_key="k1")


def test_every_call_is_audited_including_refusals(store):
    store.get_open_tickets(thread_id="t1")
    with pytest.raises(GMAOError):
        store.create_ticket(**{**TICKET, "priority": "P9"}, approved_by="tech.a", idempotency_key="k", thread_id="t1")
    store.create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k", thread_id="t1")
    store.create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k", thread_id="t1")
    outcomes = [(e["tool"], e["outcome"].split(":")[0]) for e in store.audit_log("t1")]
    assert outcomes == [("get_open_tickets", "ok"), ("create_ticket", "refused"),
                        ("create_ticket", "created"), ("create_ticket", "replayed")]


def test_conflict_refusal_is_audited(store):
    store.create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k", thread_id="t")
    with pytest.raises(GMAOError):
        store.create_ticket(**{**TICKET, "title": "Autre"}, approved_by="tech.a", idempotency_key="k", thread_id="t")
    assert store.audit_log("t")[-1]["outcome"].startswith("refused: Conflit d'idempotence")


def test_persistence_across_restarts(tmp_path):
    db = tmp_path / "gmao.sqlite"
    GMAOStore(db).create_ticket(**TICKET, approved_by="tech.a", idempotency_key="k")
    reopened = GMAOStore(db)  # pas de ré-amorçage des données de départ
    assert any(t["id"] == "INC-131" for t in reopened.get_open_tickets("FABRIC-A"))
    assert len(reopened.get_equipment_history("FABRIC-A")["tickets"]) == 2


# ── Serveur MCP ─────────────────────────────────────────────────────────────

def test_server_exposes_the_three_sprint4_tools(client):
    assert set(client.list_tools()) == {"get_equipment_history", "get_open_tickets", "create_ticket"}


def test_mcp_create_ticket_roundtrip_and_replay(client):
    args = dict(**TICKET, approved_by="tech.a", idempotency_key="k1")
    first, again = client.create_ticket(**args), client.create_ticket(**args)
    assert first["created"] is True and again["created"] is False
    assert first["ticket"]["id"] == again["ticket"]["id"]


def test_mcp_business_error_is_readable(client):
    with pytest.raises(GMAOToolError, match="Équipements connus"):
        client.get_equipment_history("CTRL-Z")
    with pytest.raises(GMAOToolError, match="approbation"):
        client.create_ticket(**TICKET, approved_by="", idempotency_key="k")


# ── Extraction déterministe ─────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("Crée un ticket pour le contrôleur B en défaut", "CTRL-B"),
    ("alarme AL-01 sur le fabric A", "FABRIC-A"),
    ("Le disque B03 a-t-il un antécédent ?", "DISK-B03"),
    ("capacité du VOL-DB", "VOL-DB"),
    ("Quels tickets sont ouverts sur la baie ?", "SAN-LAB-2000"),
    ("Ça clignote", None),
])
def test_extract_equipment(text, expected):
    assert actions.extract_equipment(text) == expected


@pytest.mark.parametrize("text, kind", [
    ("Crée un ticket pour le contrôleur B", "ticket"),
    ("Ouvre un ticket pour préparer un failover", "ticket"),  # le ticket documente, n'exécute rien
    ("Fais un failover vers le contrôleur A", "forbidden"),
    ("Modifie le zoning du fabric B", "forbidden"),
    ("Envoie un mail au fournisseur", "unsupported"),
])
def test_classify_action(text, kind):
    assert actions.classify_action(text) == kind


def test_idempotency_key_is_stable_and_content_sensitive():
    d = dict(equipment_id="CTRL-B", title="t", description="d", priority="P3", evidence=["INC-130"])
    assert actions.idempotency_key("th", d) == actions.idempotency_key("th", dict(d, warnings=["ignoré"]))
    assert actions.idempotency_key("th", d) != actions.idempotency_key("th", dict(d, priority="P1"))
    assert actions.idempotency_key("th", d) != actions.idempotency_key("autre", d)
