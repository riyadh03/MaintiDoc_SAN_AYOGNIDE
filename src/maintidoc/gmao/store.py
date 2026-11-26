"""GMAO simulée (SQLite) : équipements, tickets d'intervention et journal d'audit.

C'est le « système d'information » auquel l'agent accède UNIQUEMENT via le serveur MCP.
Les règles métier qui ne doivent pas dépendre du LLM sont appliquées ICI, côté serveur :
équipement connu, priorité valide, approbateur obligatoire, idempotence, audit.

Données initiales cohérentes avec le manuel SAN-MAN-001 (section 6, historique fictif).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRIORITIES = ("P1", "P2", "P3", "P4")
OPEN_STATUSES = ("open", "monitoring")

SEED_EQUIPMENT = [
    ("SAN-LAB-2000", "Baie SAN-LAB 2000", "baie"),
    ("CTRL-A", "Contrôleur A", "controleur"),
    ("CTRL-B", "Contrôleur B", "controleur"),
    ("FABRIC-A", "Fabric Fibre Channel A", "fabric"),
    ("FABRIC-B", "Fabric Fibre Channel B", "fabric"),
    ("DISK-B03", "Disque B03", "disque"),
    ("VOL-DB", "Volume VOL-DB", "volume"),
]

# (id, équipement, date, titre, statut) — manuel p.7, « Historique fictif d'interventions »
SEED_TICKETS = [
    ("INC-104", "FABRIC-A", "2026-09-18", "Perte temporaire de chemin HBA1 ; surveillance", "closed"),
    ("INC-117", "DISK-B03", "2026-10-02", "Pré-alerte ; examen du groupe de protection", "open"),
    ("INC-125", "VOL-DB", "2026-10-20", "Capacité à 88 % ; analyse de tendance", "monitoring"),
    ("INC-130", "CTRL-B", "2026-10-25", "Événement non reproductible ; suivi", "open"),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS equipment (
    id TEXT PRIMARY KEY, label TEXT NOT NULL, type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL REFERENCES equipment(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'P3',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    approved_by TEXT,
    evidence TEXT NOT NULL DEFAULT '[]',
    thread_id TEXT,
    idempotency_key TEXT UNIQUE,
    payload_hash TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tool TEXT NOT NULL,
    actor TEXT NOT NULL,
    thread_id TEXT,
    arguments TEXT NOT NULL,
    outcome TEXT NOT NULL
);
"""


class GMAOError(ValueError):
    """Requête refusée par une règle métier de la GMAO (message lisible par le LLM et l'humain)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class GMAOStore:
    def __init__(self, path: Path | str = ":memory:", seed: bool = True):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            if seed and not self._conn.execute("SELECT 1 FROM equipment LIMIT 1").fetchone():
                self._conn.executemany("INSERT INTO equipment VALUES (?, ?, ?)", SEED_EQUIPMENT)
                self._conn.executemany(
                    "INSERT INTO tickets (id, equipment_id, created_at, title, status, created_by) "
                    "VALUES (?, ?, ?, ?, ?, 'historique')",
                    [(t[0], t[1], t[2], t[3], t[4]) for t in SEED_TICKETS],
                )

    # ── Lecture ─────────────────────────────────────────────────────────────

    def equipment(self, equipment_id: str) -> Dict[str, Any]:
        row = self._conn.execute("SELECT * FROM equipment WHERE id = ?", (equipment_id,)).fetchone()
        if row is None:
            known = [r["id"] for r in self._conn.execute("SELECT id FROM equipment ORDER BY id")]
            raise GMAOError(f"Équipement inconnu : '{equipment_id}'. Équipements connus : {', '.join(known)}.")
        return dict(row)

    def get_equipment_history(self, equipment_id: str, *, actor: str = "maintidoc-agent",
                              thread_id: Optional[str] = None) -> Dict[str, Any]:
        args = {"equipment_id": equipment_id}
        try:
            eq = self.equipment(equipment_id)
        except GMAOError as exc:
            self._audit("get_equipment_history", actor, thread_id, args, f"refused: {exc}")
            raise
        rows = self._conn.execute(
            "SELECT * FROM tickets WHERE equipment_id = ? ORDER BY created_at DESC", (equipment_id,)
        ).fetchall()
        tickets = [self._ticket(r) for r in rows]
        self._audit("get_equipment_history", actor, thread_id, args, f"ok: {len(tickets)} ticket(s)")
        return {"equipment": eq, "tickets": tickets}

    def get_open_tickets(self, equipment_id: Optional[str] = None, *, actor: str = "maintidoc-agent",
                         thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        args = {"equipment_id": equipment_id}
        if equipment_id:
            try:
                self.equipment(equipment_id)
            except GMAOError as exc:
                self._audit("get_open_tickets", actor, thread_id, args, f"refused: {exc}")
                raise
        sql = f"SELECT * FROM tickets WHERE status IN ({','.join('?' * len(OPEN_STATUSES))})"
        params: Tuple[Any, ...] = OPEN_STATUSES
        if equipment_id:
            sql += " AND equipment_id = ?"
            params += (equipment_id,)
        tickets = [self._ticket(r) for r in self._conn.execute(sql + " ORDER BY created_at DESC", params)]
        self._audit("get_open_tickets", actor, thread_id, args, f"ok: {len(tickets)} ticket(s)")
        return tickets

    # ── Écriture (action engageante) ────────────────────────────────────────

    def create_ticket(self, *, equipment_id: str, title: str, description: str, priority: str,
                      evidence: List[str], approved_by: str, idempotency_key: str,
                      thread_id: Optional[str] = None, actor: str = "maintidoc-agent") -> Tuple[Dict[str, Any], bool]:
        """Crée un ticket APPROUVÉ. Retourne (ticket, created).

        Idempotence : la même ``idempotency_key`` avec le même contenu renvoie le ticket déjà
        créé (``created=False``) au lieu d'un doublon ; avec un contenu différent, c'est un
        conflit (refus) : une clé ne doit jamais désigner deux demandes distinctes.
        """
        payload = {"equipment_id": equipment_id, "title": title.strip(), "description": description.strip(),
                   "priority": priority, "evidence": list(evidence)}
        args = {**payload, "approved_by": approved_by, "idempotency_key": idempotency_key}

        def refuse(msg: str):
            self._audit("create_ticket", actor, thread_id, args, f"refused: {msg}")
            raise GMAOError(msg)

        if not (approved_by or "").strip():
            refuse("Création refusée : approbation humaine absente (approved_by vide).")
        if not (idempotency_key or "").strip():
            refuse("Création refusée : clé d'idempotence absente.")
        if priority not in PRIORITIES:
            refuse(f"Priorité invalide '{priority}' : attendu {', '.join(PRIORITIES)}.")
        if not payload["title"]:
            refuse("Création refusée : titre vide.")
        try:
            self.equipment(equipment_id)
        except GMAOError as exc:
            refuse(str(exc))

        digest = payload_hash(payload)
        conflict = False
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None and existing["payload_hash"] != digest:
                # Pas de raise ici : une exception annulerait la transaction, et donc l'audit
                conflict = True
            elif existing is not None:
                ticket = self._ticket(existing)
                self._audit("create_ticket", actor, thread_id, args, f"replayed: {ticket['id']}", locked=True)
                return ticket, False
            else:
                ticket_id = self._next_ticket_id()
                self._conn.execute(
                    "INSERT INTO tickets (id, equipment_id, title, description, priority, status, created_at, "
                    "created_by, approved_by, evidence, thread_id, idempotency_key, payload_hash) "
                    "VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, equipment_id, payload["title"], payload["description"], priority, _now(), actor,
                     approved_by.strip(), json.dumps(payload["evidence"], ensure_ascii=False), thread_id,
                     idempotency_key, digest),
                )
                self._audit("create_ticket", actor, thread_id, args, f"created: {ticket_id}", locked=True)
        if conflict:
            refuse("Conflit d'idempotence : cette clé correspond déjà à un ticket au contenu différent.")
        return self._ticket(self._conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()), True

    # ── Audit ──────────────────────────────────────────────────────────────

    def audit_log(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, params = "SELECT * FROM audit_log", ()
        if thread_id:
            sql, params = sql + " WHERE thread_id = ?", (thread_id,)
        return [{**dict(r), "arguments": json.loads(r["arguments"])} for r in self._conn.execute(sql + " ORDER BY id", params)]

    # ── Interne ────────────────────────────────────────────────────────────

    def _audit(self, tool: str, actor: str, thread_id: Optional[str], arguments: Dict[str, Any],
               outcome: str, locked: bool = False) -> None:
        row = (_now(), tool, actor, thread_id, json.dumps(arguments, ensure_ascii=False), outcome)
        sql = "INSERT INTO audit_log (ts, tool, actor, thread_id, arguments, outcome) VALUES (?, ?, ?, ?, ?, ?)"
        if locked:  # déjà dans la transaction de create_ticket
            self._conn.execute(sql, row)
            return
        with self._lock, self._conn:
            self._conn.execute(sql, row)

    def _next_ticket_id(self) -> str:
        nums = [int(r["id"].split("-")[1]) for r in self._conn.execute("SELECT id FROM tickets WHERE id LIKE 'INC-%'")]
        return f"INC-{max(nums, default=100) + 1}"

    @staticmethod
    def _ticket(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["evidence"] = json.loads(d.get("evidence") or "[]")
        d.pop("payload_hash", None)
        return d
