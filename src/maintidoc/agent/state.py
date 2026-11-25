"""État partagé du graphe MaintiDoc et décision de routage.

Pourquoi un état explicite : chaque nœud du graphe lit l'état et renvoie une *mise à jour
partielle*. Toute la conversation (question, clarifications, preuves, tentatives) est donc
inspectable à chaque étape, et sauvegardée par le checkpointer entre deux tours
(indispensable pour reprendre après une question de clarification).

Les résultats de recherche y sont stockés en ``dict`` (``SearchHit.model_dump()``) et non
en objets Pydantic : le checkpointer SQLite sérialise l'état, et des types primitifs
garantissent une sérialisation stable d'une version de bibliothèque à l'autre.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Intent = Literal[
    "documentation",     # réponse attendue dans le manuel SAN → chaîne RAG
    "ambiguous",         # équipement / symptôme non précisé → question de clarification
    "operational_data",  # données vivantes (tickets ouverts, historique réel) → MCP (Sprint 4)
    "action_request",    # demande d'agir (créer un ticket, failover…) → validation humaine (Sprint 4)
    "out_of_scope",      # hors maintenance SAN
]


class RouteDecision(BaseModel):
    """Sortie structurée du routeur (LLM contraint par schéma JSON)."""
    intent: Intent
    reason: str = Field(description="Justification courte du routage, pour la traçabilité.")
    search_query: str = Field(
        description="Requête de recherche en français : termes techniques et codes exacts de la question."
    )
    visual_query_en: Optional[str] = Field(
        default=None,
        description="Description ANGLAISE du schéma utile (CLIP est entraîné en anglais), ou null.",
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Question à poser à l'utilisateur si intent = ambiguous."
    )


class Clarification(TypedDict):
    question: str
    answer: str


class AgentState(TypedDict, total=False):
    # Entrée
    question: str
    clarifications: List[Clarification]

    # Routage
    intent: Intent
    route_reason: str
    search_query: str
    visual_query_en: Optional[str]
    pending_clarification: Optional[str]

    # Recherche
    retrieval_attempt: int
    text_hits: List[Dict[str, Any]]
    visual_hits: List[Dict[str, Any]]
    reranked_hits: List[Dict[str, Any]]

    # Preuves et génération
    evidence_texts: List[Dict[str, Any]]
    evidence_images: List[Dict[str, Any]]
    allowed_citations: Dict[str, Dict[str, Any]]
    generation_attempt: int
    llm_output: Dict[str, Any]
    citation_issues: List[str]

    # Sortie
    response: Dict[str, Any]
    # Journal des nœuds traversés : le réducteur ``operator.add`` CONCATÈNE au lieu de
    # remplacer, chaque nœud n'a qu'à renvoyer son propre nom.
    trace: Annotated[List[str], operator.add]


def effective_question(state: AgentState) -> str:
    """Question enrichie des précisions apportées par l'utilisateur."""
    parts = [state["question"]]
    for c in state.get("clarifications") or []:
        parts.append(f"(Précision demandée : {c['question']} → Réponse : {c['answer']})")
    return "\n".join(parts)
