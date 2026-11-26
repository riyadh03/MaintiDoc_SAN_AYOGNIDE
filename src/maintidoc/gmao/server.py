"""Serveur MCP de la GMAO MaintiDoc : 3 outils normalisés, règles métier appliquées côté serveur.

Lancement (transport stdio, c'est ce que fait l'agent automatiquement) :
    python -m maintidoc.gmao.server

Pourquoi MCP : l'agent ne connaît que des outils décrits par un schéma JSON (découverts via
``list_tools``), pas la base. Remplacer la GMAO simulée par une vraie (GLPI, Maximo…) ne
change que ce serveur. Et une règle codée ici (approbateur obligatoire, idempotence) ne peut
pas être contournée par un LLM trompé, contrairement à une consigne dans un prompt.
"""
from __future__ import annotations

from typing import List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from maintidoc.config import GMAO_DB
from maintidoc.gmao.store import GMAOError, GMAOStore


class Ticket(BaseModel):
    id: str
    equipment_id: str
    title: str
    description: str = ""
    priority: str = "P3"
    status: str
    created_at: str
    created_by: str
    approved_by: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None
    idempotency_key: Optional[str] = None


class Equipment(BaseModel):
    id: str
    label: str
    type: str


class EquipmentHistory(BaseModel):
    equipment: Equipment
    tickets: List[Ticket]


class TicketList(BaseModel):
    tickets: List[Ticket]


class TicketResult(BaseModel):
    ticket: Ticket
    created: bool = Field(description="False si la clé d'idempotence avait déjà servi : ticket existant renvoyé.")


INSTRUCTIONS = (
    "GMAO de la baie SAN-LAB 2000. Lecture libre de l'historique et des tickets ouverts. "
    "create_ticket est une action engageante : elle n'est acceptée qu'avec un approbateur humain "
    "(approved_by) et une clé d'idempotence."
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
_CREATE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def build_server(store: GMAOStore) -> MCPServer:
    # Les refus métier sont journalisés dans audit_log ; inutile de les répéter en INFO sur stderr
    mcp = MCPServer("maintidoc-gmao", instructions=INSTRUCTIONS, version="0.4.0", log_level="WARNING")

    @mcp.tool(annotations=_READ_ONLY)
    def get_equipment_history(equipment_id: str, thread_id: Optional[str] = None) -> EquipmentHistory:
        """Historique complet des interventions (tickets ouverts et clos) d'un équipement.

        equipment_id : SAN-LAB-2000, CTRL-A, CTRL-B, FABRIC-A, FABRIC-B, DISK-B03, VOL-DB…
        """
        try:
            return EquipmentHistory.model_validate(store.get_equipment_history(equipment_id, thread_id=thread_id))
        except GMAOError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(annotations=_READ_ONLY)
    def get_open_tickets(equipment_id: Optional[str] = None, thread_id: Optional[str] = None) -> TicketList:
        """Tickets ouverts ou en surveillance, pour un équipement ou pour toute la baie (equipment_id omis)."""
        try:
            return TicketList(tickets=store.get_open_tickets(equipment_id, thread_id=thread_id))
        except GMAOError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(annotations=_CREATE)
    def create_ticket(
        equipment_id: str,
        title: str,
        description: str,
        priority: str,
        approved_by: str,
        idempotency_key: str,
        evidence: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> TicketResult:
        """Crée un ticket d'intervention APPROUVÉ par un humain.

        priority : P1 (critique) à P4 (faible). evidence : références justificatives
        (citations du manuel, tickets liés). Rejouer la même idempotency_key renvoie le ticket
        existant sans doublon.
        """
        try:
            ticket, created = store.create_ticket(
                equipment_id=equipment_id, title=title, description=description, priority=priority,
                evidence=evidence or [], approved_by=approved_by, idempotency_key=idempotency_key,
                thread_id=thread_id,
            )
        except GMAOError as exc:
            raise ToolError(str(exc)) from exc
        return TicketResult(ticket=Ticket.model_validate(ticket), created=created)

    return mcp


def main() -> None:
    build_server(GMAOStore(GMAO_DB)).run("stdio")


if __name__ == "__main__":
    main()
