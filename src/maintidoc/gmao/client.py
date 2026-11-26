"""Client MCP synchrone de la GMAO, utilisé par les nœuds du graphe LangGraph.

Le graphe est synchrone, le SDK MCP est asynchrone. Chaque appel s'exécute avec
``anyio.run`` dans un thread dédié : cela fonctionne aussi dans Jupyter, où une boucle
asyncio tourne déjà dans le thread principal (``asyncio.run`` y est interdit).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import anyio
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from maintidoc.config import GMAO_DB


class GMAOToolError(RuntimeError):
    """L'outil MCP a répondu par une erreur (règle métier, équipement inconnu, serveur absent…)."""


class GMAOClient(Protocol):
    def get_equipment_history(self, equipment_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]: ...

    def get_open_tickets(self, equipment_id: Optional[str] = None, thread_id: Optional[str] = None) -> Dict[str, Any]: ...

    def create_ticket(self, **kwargs: Any) -> Dict[str, Any]: ...


class MCPGMAOClient:
    """Appelle les outils du serveur MCP GMAO.

    ``target`` : ``StdioServerParameters`` (production : sous-processus) ou l'objet serveur
    lui-même (tests : transport en mémoire).
    """

    def __init__(self, target: Any, timeout_s: float = 60.0):
        self.target = target
        self.timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-gmao")

    @classmethod
    def stdio(cls, db_path: Path | str = GMAO_DB, **kw: Any) -> "MCPGMAOClient":
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "maintidoc.gmao.server"],
            env={**os.environ, "MAINTIDOC_GMAO_DB": str(db_path), "PYTHONIOENCODING": "utf-8"},
        )
        return cls(params, **kw)

    # ── Outils ──────────────────────────────────────────────────────────────

    def list_tools(self) -> List[str]:
        async def go():
            async with Client(self.target) as client:
                return [t.name for t in (await client.list_tools()).tools]
        return self._run(go)

    def get_equipment_history(self, equipment_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        return self.call("get_equipment_history", {"equipment_id": equipment_id, "thread_id": thread_id})

    def get_open_tickets(self, equipment_id: Optional[str] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        return self.call("get_open_tickets", {"equipment_id": equipment_id, "thread_id": thread_id})

    def create_ticket(self, **kwargs: Any) -> Dict[str, Any]:
        return self.call("create_ticket", kwargs)

    # ── Transport ───────────────────────────────────────────────────────────

    def call(self, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        arguments = {k: v for k, v in arguments.items() if v is not None}

        async def go():
            async with Client(self.target, read_timeout_seconds=self.timeout_s) as client:
                return await client.call_tool(tool, arguments)

        result = self._run(go)
        if result.is_error:
            text = " ".join(c.text for c in result.content if isinstance(c, TextContent))
            raise GMAOToolError(text or f"Échec de l'outil {tool}")
        return result.structured_content or {}

    def _run(self, coro_fn):
        try:
            return self._pool.submit(anyio.run, coro_fn).result(timeout=self.timeout_s + 30)
        except GMAOToolError:
            raise
        except Exception as exc:  # serveur absent, timeout, protocole…
            raise GMAOToolError(f"Serveur MCP GMAO injoignable : {exc}") from exc
