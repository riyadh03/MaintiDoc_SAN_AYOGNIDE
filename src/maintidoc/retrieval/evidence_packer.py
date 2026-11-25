"""Evidence Packer — Sélection budgétée des preuves et construction des citations autorisées.

Pourquoi ce module existe :
  - Les LLMs ont une fenêtre de contexte limitée (num_ctx = 8 192 tokens pour Qwen).
  - Il faut sélectionner les passages les plus pertinents SANS dépasser le budget.
  - Les images consomment aussi des tokens : on leur réserve un budget séparé pour
    qu'un long contexte textuel ne puisse jamais évincer le schéma technique.
  - Les citations doivent être VALIDÉES : le LLM ne doit référencer que des
    éléments réellement présents dans le contexte fourni.

Workflow :
  1. Reçoit les hits textuels reclassés (CrossEncoder) + hits visuels (CLIP).
  2. Remplit le contexte dans l'ordre de pertinence. Un passage trop long pour le
     budget restant est SAUTÉ (``continue``) : les passages suivants, plus courts,
     peuvent encore entrer.
  3. Construit ``allowed_citations`` indexé par l'identifiant canonique
     ``<document_id>:p<page>:<element_id>`` (voir ``schemas.make_citation_id``).
  4. Ce dictionnaire est la référence de ``validate_citations``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from maintidoc.config import DOC_ID
from maintidoc.schemas import SearchHit, make_citation_id

logger = logging.getLogger(__name__)

# Estimation grossière : 1 token ≈ 4 caractères (convention LLM, pessimiste pour le français)
_CHARS_PER_TOKEN = 4

# Coût estimé d'une image dans le contexte VLM (en tokens équivalents)
_TOKENS_PER_IMAGE = 500


def estimate_tokens(text: str) -> int:
    """Estimation du nombre de tokens d'un texte (au moins 1)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


class EvidencePacker:
    """Sélectionne et emballe les preuves dans le budget token du LLM.

    Args:
        max_tokens : Budget total en tokens, images comprises (défaut : 6 000,
                     marge laissée au prompt système et à la réponse).
        max_images : Nombre maximum d'images à inclure (défaut : 1). Leur coût est
                     réservé AVANT le remplissage textuel.

    Usage :
        packer = EvidencePacker(max_tokens=6000, max_images=1)
        texts, images, citations = packer.pack(text_hits, visual_hits)
    """

    def __init__(self, max_tokens: int = 6_000, max_images: int = 1):
        if max_tokens <= 0:
            raise ValueError("max_tokens doit être strictement positif.")
        if max_images < 0:
            raise ValueError("max_images doit être positif ou nul.")
        if max_images * _TOKENS_PER_IMAGE > max_tokens:
            raise ValueError(
                f"Budget insuffisant : {max_images} image(s) × {_TOKENS_PER_IMAGE} tokens "
                f"> max_tokens={max_tokens}."
            )
        self.max_tokens = max_tokens
        self.max_images = max_images

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def pack(
        self,
        text_hits: List[SearchHit],
        visual_hits: List[SearchHit],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Sélectionne les preuves dans le budget et construit les citations autorisées.

        Args:
            text_hits   : SearchHit textuels/tabulaires, triés par pertinence décroissante.
            visual_hits : SearchHit visuels (CLIP), triés par score décroissant.

        Returns:
            (selected_texts, selected_images, allowed_citations)

            selected_texts : [{"citation_id", "document_id", "element_id", "page",
                               "modality", "content", "score", "rank"}, …]
            selected_images: [{"citation_id", "document_id", "element_id", "page",
                               "figure_id", "image_path", "score"}, …]
            allowed_citations: {citation_id → {"page", "element_id", "modality", "snippet"}}
        """
        # 1. Réserver le budget images en premier : le schéma ne doit pas être évincé
        images = [h for h in visual_hits if h.metadata.get("image_path")][: self.max_images]
        image_budget = len(images) * _TOKENS_PER_IMAGE
        text_budget = self.max_tokens - image_budget

        selected_texts: List[Dict[str, Any]] = []
        allowed_citations: Dict[str, Dict[str, Any]] = {}
        skipped: List[str] = []
        used = 0

        # 2. Remplissage glouton des textes dans l'ordre de pertinence
        for hit in text_hits:
            content = (hit.metadata.get("content") or "").strip()
            if not content:
                continue
            cost = estimate_tokens(content)
            if used + cost > text_budget:
                # On saute ce passage mais on continue : un passage suivant plus court
                # peut encore tenir dans le budget restant.
                skipped.append(f"{hit.element_id} ({cost} tokens)")
                continue

            document_id = hit.metadata.get("document_id") or DOC_ID
            citation_id = make_citation_id(document_id, hit.page, hit.element_id)
            if citation_id in allowed_citations:
                continue  # doublon (même élément remonté deux fois)

            used += cost
            selected_texts.append(
                {
                    "citation_id": citation_id,
                    "document_id": document_id,
                    "element_id": hit.element_id,
                    "page": hit.page,
                    "modality": hit.modality,
                    "content": content,
                    "score": hit.score,
                    "rank": hit.rank,
                }
            )
            allowed_citations[citation_id] = {
                "page": hit.page,
                "element_id": hit.element_id,
                "modality": hit.modality,
                "snippet": content[:80].replace("\n", " ").strip(),
            }

        if skipped:
            logger.info("EvidencePacker : passages sautés faute de budget : %s", ", ".join(skipped))

        # 3. Images (budget déjà réservé)
        selected_images: List[Dict[str, Any]] = []
        for hit in images:
            document_id = hit.metadata.get("document_id") or DOC_ID
            citation_id = make_citation_id(document_id, hit.page, hit.element_id)
            selected_images.append(
                {
                    "citation_id": citation_id,
                    "document_id": document_id,
                    "element_id": hit.element_id,
                    "page": hit.page,
                    "figure_id": hit.metadata.get("figure_id"),
                    "image_path": hit.metadata["image_path"],
                    "score": hit.score,
                }
            )
            allowed_citations[citation_id] = {
                "page": hit.page,
                "element_id": hit.element_id,
                "modality": hit.modality,
                "snippet": f"[schéma technique {hit.metadata.get('figure_id') or ''} page {hit.page}]".replace("  ", " "),
            }

        logger.info(
            "EvidencePacker : %d textes, %d images | %d/%d tokens utilisés.",
            len(selected_texts),
            len(selected_images),
            used + image_budget,
            self.max_tokens,
        )
        return selected_texts, selected_images, allowed_citations
