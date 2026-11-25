"""Cross-Encoder Reranker — Reclassement sémantique fin post-retrieval.

Pourquoi ce module existe :
  - Le retrieval hybride (Dense + BM25 + RRF) maximise le RAPPEL : il remonte
    beaucoup de candidats, mais leur ordre est imparfait.
  - Un Cross-Encoder lit ENSEMBLE la question et chaque passage et produit un
    score de pertinence : la PRÉCISION est maximisée en tête de liste.

Bi-Encoder vs Cross-Encoder :
  - Bi-Encoder (BGE-M3) : encode question et doc SÉPARÉMENT → vecteurs pré-calculables,
    très rapide, mais l'interaction mot-à-mot question/doc est perdue.
  - Cross-Encoder        : encode la paire (question, doc) ENSEMBLE → l'attention croise
    chaque mot de la question avec chaque mot du passage. Précis mais coûteux
    (un passage complet du transformer PAR paire) → réservé à quelques dizaines de candidats.

Choix du modèle : BAAI/bge-reranker-v2-m3
  - Multilingue (initialisé depuis BGE-M3, comme nos embeddings) → robuste en français.
  - bge-reranker-base, l'alternative plus légère, est entraîné surtout zh/en.
  - Coût : ~570 Mo, plus lent sur CPU ; acceptable pour ~10 candidats.

Calibration du score :
  - Le modèle sort un logit ; on applique explicitement une sigmoïde → score dans [0, 1].
  - Un score stable dans [0, 1] permet un SEUIL (``min_score``) : sous ce seuil un
    passage n'est pas une preuve. Si plus rien ne passe, l'orchestrateur doit
    répondre ``insufficient_evidence`` plutôt que laisser le LLM extrapoler.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from maintidoc.config import RERANKER_MODEL
from maintidoc.schemas import SearchHit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker sémantique basé sur un cross-encoder multilingue.

    Args:
        model_name : Modèle HuggingFace (défaut : ``config.RERANKER_MODEL``).
        device     : 'cpu', 'cuda' ou None (détection automatique).
        max_length : Longueur max (tokens) d'une paire question+passage. Au-delà, le
                     passage est TRONQUÉ et sa fin ignorée : 1024 couvre nos chunks
                     (page la plus longue ≈ 2 500 caractères).
        model      : Objet exposant ``predict(pairs) -> scores`` déjà construit.
                     Sert à l'injection de dépendance (tests sans téléchargement).

    Usage :
        reranker = CrossEncoderReranker()
        top = reranker.rerank("Que signifie le voyant orange ?", hits, top_k=4, min_score=0.05)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 1024,
        model: Any = None,
    ):
        self._model_name = model_name or RERANKER_MODEL
        self._device = device
        self._max_length = max_length
        self._model = model  # None → chargement paresseux au premier rerank()

    # ------------------------------------------------------------------
    # Chargement paresseux du modèle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Charge le CrossEncoder en mémoire (une seule fois)."""
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers manquant. Installer avec : uv add sentence-transformers"
            ) from exc

        logger.info("Chargement du cross-encoder %s …", self._model_name)
        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=self._max_length,
            activation_fn=torch.nn.Sigmoid(),  # logit → probabilité [0, 1], explicite
        )
        logger.info("Cross-encoder prêt.")

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def score(self, question: str, hits: List[SearchHit]) -> List[float]:
        """Score de pertinence [0, 1] de chaque hit, dans l'ordre d'entrée."""
        if not hits:
            return []
        self._load()
        pairs = [(question, h.metadata.get("content", "")) for h in hits]
        return [float(s) for s in self._model.predict(pairs)]

    def rerank(
        self,
        question: str,
        hits: List[SearchHit],
        top_k: int = 4,
        min_score: Optional[float] = None,
    ) -> List[SearchHit]:
        """Reclasse les SearchHit par pertinence croisée (question, passage).

        Args:
            question  : Question de l'utilisateur.
            hits      : Candidats produits par ``HybridRetriever``.
            top_k     : Nombre maximum de passages retournés.
            min_score : Seuil [0, 1] ; les passages en dessous sont écartés.
                        None = pas de filtrage. Liste vide en sortie = preuves insuffisantes.

        Returns:
            SearchHit reclassés : ``score`` = score cross-encoder, et la trace
            (``reranker_score``, ``pre_rerank_score``, ``pre_rerank_rank``) dans ``metadata``.
        """
        scores = self.score(question, hits)
        # sorted() est stable : à score égal, l'ordre du retrieval est conservé
        scored = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)

        if min_score is not None:
            kept = [(s, h) for s, h in scored if s >= min_score]
            if len(kept) < len(scored):
                logger.info(
                    "Reranker : %d/%d passages sous le seuil %.3f écartés.",
                    len(scored) - len(kept), len(scored), min_score,
                )
            scored = kept

        return [
            SearchHit(
                element_id=hit.element_id,
                page=hit.page,
                modality=hit.modality,
                score=round(ce_score, 6),
                rank=new_rank,
                metadata={
                    **hit.metadata,
                    "reranker_score": round(ce_score, 6),
                    "pre_rerank_score": hit.score,
                    "pre_rerank_rank": hit.rank,
                },
            )
            for new_rank, (ce_score, hit) in enumerate(scored[:top_k], 1)
        ]

    def rerank_with_scores(self, question: str, hits: List[SearchHit]) -> List[Tuple[float, SearchHit]]:
        """Variante sans coupe ni seuil, retournant (score, hit) pour analyse / débogage."""
        reranked = self.rerank(question, hits, top_k=len(hits))
        return [(h.score, h) for h in reranked]
