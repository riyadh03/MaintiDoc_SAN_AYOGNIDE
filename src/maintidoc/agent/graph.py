"""Graphe d'état MaintiDoc (LangGraph) : routage, boucles de contrôle bornées, clarification.

Topologie (graphe EXPLICITE : le LLM classe et rédige, le code Python décide des transitions) :

    START → classify ─┬─ documentation ───► retrieve → rerank ─┬─ preuves ──► pack → generate → validate
                      │                        ▲                └─ aucune ──► rewrite_query (≤1) ┐
                      │                        └───────────────────────────────────────────────┘
                      ├─ ambiguous ───────► clarify (interrupt) ──► classify
                      ├─ operational_data ► respond_tool_unavailable ─► END   (MCP : Sprint 4)
                      ├─ action_request ──► respond_action_refused ───► END   (HITL : Sprint 4)
                      └─ out_of_scope ────► respond_out_of_scope ─────► END

    validate ─┬─ citations valides ───────────────► finalize → END
              ├─ citations invalides (≤1 retry) ──► generate (avec le retour du validateur)
              ├─ citations invalides (épuisé) ────► reject → END
              └─ LLM demande une précision ───────► clarify

Pourquoi un graphe explicite plutôt qu'un agent ReAct : chaque transition est du code
testable, le nombre d'appels LLM est borné et prévisible, et un modèle local n'a pas à
fiabiliser des appels d'outils libres. Toutes les boucles ont un plafond (config.AGENT_MAX_*).
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from maintidoc.agent.llm import Generator, Router
from maintidoc.agent.state import AgentState, RouteDecision, effective_question
from maintidoc.config import (
    AGENT_CHECKPOINT_DB,
    AGENT_MAX_CLARIFICATIONS,
    AGENT_MAX_GENERATION_ATTEMPTS,
    AGENT_MAX_RETRIEVAL_ATTEMPTS,
    AGENT_RERANK_TOP_K,
    RERANKER_MIN_SCORE,
)
from maintidoc.generation.ollama_client import validate_citations
from maintidoc.schemas import SearchHit

logger = logging.getLogger(__name__)

DEFAULT_CLARIFICATION = (
    "Pouvez-vous préciser l'équipement concerné (contrôleur, disque, fabric, voyant) "
    "et le code d'alarme ou le symptôme observé ?"
)

CANNED_RESPONSES = {
    "out_of_scope": (
        "Je suis MaintiDoc, assistant documentaire de maintenance SAN : cette question sort "
        "de mon périmètre."
    ),
    "tool_unavailable": (
        "Cette question porte sur des données opérationnelles en temps réel (tickets ouverts, "
        "historique d'équipement). Leur consultation n'est pas encore connectée ; je peux "
        "en revanche répondre à partir du manuel de maintenance."
    ),
    "action_refused": (
        "MaintiDoc n'exécute aucune action technique (ticket, failover, zoning, suppression). "
        "Toute action engageante nécessitera une validation humaine explicite."
    ),
}


class MaintiDocGraph:
    """Construit les nœuds du graphe à partir de dépendances injectées."""

    def __init__(self, retriever, reranker, packer, router: Router, generator: Generator):
        self.retriever = retriever
        self.reranker = reranker
        self.packer = packer
        self.router = router
        self.generator = generator

    # ─────────────────────────────────────────────────────────────────────
    # Routage
    # ─────────────────────────────────────────────────────────────────────

    def classify(self, state: AgentState) -> Dict[str, Any]:
        question = effective_question(state)
        try:
            route = self.router.classify(question)
        except Exception as exc:
            # Repli sûr : le chemin documentaire reste ancré dans les preuves et cité
            logger.warning("Routeur indisponible (%s) : repli sur 'documentation'.", exc)
            route = RouteDecision(intent="documentation", reason=f"router_error: {exc}", search_query=state["question"])

        intent = route.intent
        n_clar = len(state.get("clarifications") or [])
        if intent == "ambiguous" and n_clar >= AGENT_MAX_CLARIFICATIONS:
            intent = "documentation"  # plafond atteint : on cherche avec ce qu'on a

        return {
            "intent": intent,
            "route_reason": route.reason,
            "search_query": route.search_query.strip() or state["question"],
            "visual_query_en": route.visual_query_en,
            "pending_clarification": route.clarification_question,
            "trace": [f"classify:{intent}"],
        }

    @staticmethod
    def route_after_classify(state: AgentState) -> str:
        return {
            "documentation": "retrieve",
            "ambiguous": "clarify",
            "operational_data": "respond_tool_unavailable",
            "action_request": "respond_action_refused",
            "out_of_scope": "respond_out_of_scope",
        }[state["intent"]]

    # ─────────────────────────────────────────────────────────────────────
    # Clarification (pause du graphe ; reprise via Command(resume=réponse))
    # ─────────────────────────────────────────────────────────────────────

    def clarify(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("pending_clarification") or DEFAULT_CLARIFICATION
        # interrupt() suspend l'exécution et sauvegarde l'état dans le checkpointer.
        # À la reprise, le nœud est RÉ-EXÉCUTÉ depuis le début et interrupt() renvoie la
        # réponse de l'utilisateur : tout ce qui précède doit donc être sans effet de bord.
        answer = interrupt({"type": "clarification", "question": question})
        return {
            "clarifications": [*(state.get("clarifications") or []), {"question": question, "answer": str(answer)}],
            "pending_clarification": None,
            "retrieval_attempt": 0,
            "generation_attempt": 0,
            "trace": ["clarify"],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Recherche → reranking (boucle : 1 reformulation si aucune preuve)
    # ─────────────────────────────────────────────────────────────────────

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        attempt = state.get("retrieval_attempt", 0)
        widen = attempt > 0  # 2e passage : fenêtre de candidats élargie
        text_hits, visual_hits = self.retriever.retrieve(
            question=state["search_query"],
            visual_query_en=state.get("visual_query_en"),
            dense_top_k=20 if widen else 10,
            bm25_top_k=20 if widen else 10,
            fused_top_n=12 if widen else 6,
        )
        return {
            "text_hits": [h.model_dump() for h in text_hits],
            "visual_hits": [h.model_dump() for h in visual_hits],
            "retrieval_attempt": attempt + 1,
            "trace": [f"retrieve:{attempt + 1}"],
        }

    def rerank(self, state: AgentState) -> Dict[str, Any]:
        hits = [SearchHit(**h) for h in state.get("text_hits") or []]
        reranked = self.reranker.rerank(
            effective_question(state), hits, top_k=AGENT_RERANK_TOP_K, min_score=RERANKER_MIN_SCORE
        ) if hits else []
        return {"reranked_hits": [h.model_dump() for h in reranked], "trace": [f"rerank:{len(reranked)}"]}

    @staticmethod
    def route_after_rerank(state: AgentState) -> str:
        if state.get("reranked_hits"):
            return "pack"
        if state.get("retrieval_attempt", 0) < AGENT_MAX_RETRIEVAL_ATTEMPTS:
            return "rewrite_query"
        return "insufficient_evidence"

    def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        # Reformulation déterministe : la requête « mots-clés » du routeur n'a rien donné,
        # on repart de la question complète (avec ses précisions) sur une fenêtre élargie.
        return {"search_query": effective_question(state), "trace": ["rewrite_query"]}

    # ─────────────────────────────────────────────────────────────────────
    # Preuves → génération → validation (boucle : 1 correction si citations invalides)
    # ─────────────────────────────────────────────────────────────────────

    def pack(self, state: AgentState) -> Dict[str, Any]:
        texts, images, allowed = self.packer.pack(
            [SearchHit(**h) for h in state["reranked_hits"]],
            [SearchHit(**h) for h in state.get("visual_hits") or []],
        )
        return {
            "evidence_texts": texts,
            "evidence_images": images,
            "allowed_citations": allowed,
            "generation_attempt": 0,
            "citation_issues": [],
            "trace": [f"pack:{len(texts)}t+{len(images)}i"],
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        attempt = state.get("generation_attempt", 0)
        feedback = "\n".join(state.get("citation_issues") or []) or None
        try:
            output = self.generator.generate(
                effective_question(state), state["evidence_texts"], state["evidence_images"], feedback=feedback
            )
        except Exception as exc:
            logger.warning("Génération échouée (tentative %d) : %s", attempt + 1, exc)
            output = {"_error": f"Génération impossible : {exc}"}
        return {"llm_output": output, "generation_attempt": attempt + 1, "trace": [f"generate:{attempt + 1}"]}

    def validate(self, state: AgentState) -> Dict[str, Any]:
        output = state["llm_output"]
        if "_error" in output:
            issues = [output["_error"]]
        else:
            issues = validate_citations(output, state.get("allowed_citations") or {})
        update: Dict[str, Any] = {"citation_issues": issues, "trace": [f"validate:{'ok' if not issues else 'ko'}"]}
        if not issues and output.get("status") == "needs_clarification":
            update["pending_clarification"] = output.get("clarification_question")
        return update

    @staticmethod
    def route_after_validate(state: AgentState) -> str:
        if state.get("citation_issues"):
            if state.get("generation_attempt", 0) < AGENT_MAX_GENERATION_ATTEMPTS:
                return "generate"
            return "reject"
        if (
            state["llm_output"].get("status") == "needs_clarification"
            and len(state.get("clarifications") or []) < AGENT_MAX_CLARIFICATIONS
        ):
            return "clarify"
        return "finalize"

    # ─────────────────────────────────────────────────────────────────────
    # Nœuds terminaux
    # ─────────────────────────────────────────────────────────────────────

    def finalize(self, state: AgentState) -> Dict[str, Any]:
        out = dict(state["llm_output"])
        uncertainties = list(out.get("uncertainties") or [])
        if out.get("status") == "needs_clarification":  # plafond de clarifications atteint
            out["status"] = "insufficient_evidence"
            uncertainties.append("Nombre maximal de clarifications atteint.")
        allowed = state.get("allowed_citations") or {}
        citations = [str(c).strip().strip("[]").removeprefix("SOURCE").strip() for c in out.get("citations") or []]
        return {
            "response": _response(
                state,
                status=out.get("status", "answered"),
                answer=out.get("answer", ""),
                citations=citations,
                sources=[{"citation_id": c, **allowed[c]} for c in citations if c in allowed],
                uncertainties=uncertainties,
            ),
            "trace": ["finalize"],
        }

    def reject(self, state: AgentState) -> Dict[str, Any]:
        return {
            "response": _response(
                state,
                status="insufficient_evidence",
                answer="Je ne peux pas fournir de réponse vérifiable à partir des preuves disponibles.",
                uncertainties=["Réponse rejetée par la validation des citations.", *(state.get("citation_issues") or [])],
            ),
            "trace": ["reject"],
        }

    def insufficient_evidence(self, state: AgentState) -> Dict[str, Any]:
        return {
            "response": _response(
                state,
                status="insufficient_evidence",
                answer="Le manuel ne contient pas de passage suffisamment pertinent pour répondre.",
                uncertainties=["Aucune preuve au-dessus du seuil du reranker, même après reformulation."],
            ),
            "trace": ["insufficient_evidence"],
        }

    def _canned(self, status: str):
        def node(state: AgentState) -> Dict[str, Any]:
            return {"response": _response(state, status=status, answer=CANNED_RESPONSES[status]), "trace": [status]}
        return node

    # ─────────────────────────────────────────────────────────────────────
    # Assemblage
    # ─────────────────────────────────────────────────────────────────────

    def build(self) -> StateGraph:
        g = StateGraph(AgentState)
        for name in ("classify", "clarify", "retrieve", "rerank", "rewrite_query", "pack",
                     "generate", "validate", "finalize", "reject", "insufficient_evidence"):
            g.add_node(name, getattr(self, name))
        g.add_node("respond_out_of_scope", self._canned("out_of_scope"))
        g.add_node("respond_tool_unavailable", self._canned("tool_unavailable"))
        g.add_node("respond_action_refused", self._canned("action_refused"))

        # Les destinations possibles sont déclarées (path_map) : LangGraph ne peut pas les
        # déduire d'une fonction Python, et sans elles le diagramme exporté est incomplet.
        g.add_edge(START, "classify")
        g.add_conditional_edges("classify", self.route_after_classify, [
            "retrieve", "clarify", "respond_tool_unavailable", "respond_action_refused", "respond_out_of_scope",
        ])
        g.add_edge("clarify", "classify")
        g.add_edge("retrieve", "rerank")
        g.add_conditional_edges("rerank", self.route_after_rerank, ["pack", "rewrite_query", "insufficient_evidence"])
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("pack", "generate")
        g.add_edge("generate", "validate")
        g.add_conditional_edges("validate", self.route_after_validate, ["generate", "reject", "clarify", "finalize"])
        for terminal in ("finalize", "reject", "insufficient_evidence",
                         "respond_out_of_scope", "respond_tool_unavailable", "respond_action_refused"):
            g.add_edge(terminal, END)
        return g


def _response(state: AgentState, *, status: str, answer: str, citations=None, sources=None, uncertainties=None) -> Dict[str, Any]:
    return {
        "status": status,
        "answer": answer,
        "citations": citations or [],
        "sources": sources or [],
        "uncertainties": uncertainties or [],
        "intent": state.get("intent"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Façade : API simple pour le notebook, l'application web et les canaux de chat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentTurn:
    """Résultat d'un tour : réponse finale, ou question de clarification en attente."""
    thread_id: str
    status: str  # answered | insufficient_evidence | needs_clarification | out_of_scope | tool_unavailable | action_refused
    answer: str = ""
    clarification_question: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    trace: List[str] = field(default_factory=list)


def sqlite_checkpointer(path: Path = AGENT_CHECKPOINT_DB):
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False : le serveur web (Sprint 7) appellera l'agent depuis plusieurs threads
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


class MaintiDocAgent:
    """Point d'entrée conversationnel.

    Usage :
        agent = MaintiDocAgent(retriever, reranker, packer, OllamaRouter(), OllamaGenerator())
        turn = agent.ask("Le voyant clignote, que faire ?")
        if turn.status == "needs_clarification":
            turn = agent.reply(turn.thread_id, "Voyant orange, alarme AL-01")
    """

    def __init__(self, retriever, reranker, packer, router: Router, generator: Generator, checkpointer=None):
        self.checkpointer = checkpointer if checkpointer is not None else sqlite_checkpointer()
        self.graph = MaintiDocGraph(retriever, reranker, packer, router, generator).build().compile(
            checkpointer=self.checkpointer
        )

    def ask(self, question: str, thread_id: Optional[str] = None) -> AgentTurn:
        if not question or not question.strip():
            raise ValueError("La question est vide.")
        thread_id = thread_id or str(uuid.uuid4())
        initial: AgentState = {
            "question": question.strip(),
            "clarifications": [],
            "retrieval_attempt": 0,
            "generation_attempt": 0,
            "citation_issues": [],
            "response": {},
        }
        return self._run(initial, thread_id)

    def reply(self, thread_id: str, answer: str) -> AgentTurn:
        """Répond à la question de clarification en attente et reprend le graphe."""
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot.interrupts:
            raise ValueError(f"Aucune clarification en attente pour la conversation {thread_id}.")
        return self._run(Command(resume=answer), thread_id)

    @classmethod
    def from_indexes(cls, qdrant_store, embeddings_mgr, bm25_index, checkpointer=None) -> "MaintiDocAgent":
        """Assemble l'agent de production à partir des index déjà construits (notebook, app)."""
        from maintidoc.agent.llm import OllamaGenerator, OllamaRouter
        from maintidoc.retrieval import CrossEncoderReranker, EvidencePacker, HybridRetriever

        return cls(
            HybridRetriever(qdrant_store, embeddings_mgr, bm25_index),
            CrossEncoderReranker(),
            EvidencePacker(),
            OllamaRouter(),
            OllamaGenerator(),
            checkpointer=checkpointer,
        )

    @staticmethod
    def _config(thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _run(self, payload: Any, thread_id: str) -> AgentTurn:
        result = self.graph.invoke(payload, self._config(thread_id))
        trace = result.get("trace", [])
        if result.get("__interrupt__"):
            value = result["__interrupt__"][0].value
            return AgentTurn(
                thread_id=thread_id,
                status="needs_clarification",
                clarification_question=value.get("question"),
                intent=result.get("intent"),
                trace=trace,
            )
        r = result["response"]
        return AgentTurn(
            thread_id=thread_id,
            status=r["status"],
            answer=r["answer"],
            citations=r["citations"],
            sources=r["sources"],
            uncertainties=r["uncertainties"],
            intent=r["intent"],
            trace=trace,
        )
