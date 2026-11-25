"""Recherche Hybride Multimodale — Dense (Qdrant/BGE-M3) + Sparse (BM25) fusionnés par RRF.

Pourquoi ce module existe :
  - BGE-M3 (dense) : excellent pour le sens sémantique et les paraphrases.
  - BM25 (sparse)  : excellent pour les termes exacts, codes techniques (LED-03, PSU-2, ERR_SAN_404).
  - RRF            : combine leurs classements indépendants sans avoir à normaliser leurs scores.

Workflow :
  1. Construction de l'index BM25 à partir des chunks textuels (textes + tableaux).
  2. Interrogation parallèle dense (Qdrant) et sparse (BM25) pour une requête donnée.
  3. Fusion des classements par RRF → Top-N candidats unifiés.
  4. Ajout des hits visuels CLIP (Qdrant maintidoc_visual) au dossier de preuves.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from maintidoc.schemas import TextRecord, TableRecord, SearchHit, FusionResult


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation légère (sans dépendance lourde : pas de NLTK / spaCy requis)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS_FR = {
    "le","la","les","de","du","des","un","une","et","en","au","aux","ce",
    "se","sa","son","ses","à","que","qui","pour","sur","par","dans","est",
    "il","elle","ils","elles","je","tu","nous","vous","mais","ou","car",
    "donc","ni","or","que","quoi","dont","où","plus","pas","ne","si","tout",
}

def _tokenize(text: str) -> List[str]:
    """Tokenisation simple : minuscules, suppression ponctuation, retrait stopwords."""
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_\-]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS_FR and len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Index Builder
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """Index BM25 en mémoire construit sur les chunks textuels et tableaux extraits."""

    def __init__(self):
        self._bm25 = None
        self._corpus_meta: List[Dict[str, Any]] = []  # [{element_id, page, modality, content}]

    def build(
        self,
        text_records: List[TextRecord],
        table_records: List[TableRecord],
    ) -> None:
        """Construit l'index BM25 à partir des enregistrements textuels et tabulaires."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("Installer rank-bm25 : pip install rank-bm25") from e

        self._corpus_meta = []
        tokenized_corpus = []

        for rec in text_records:
            tokens = _tokenize(rec.text)
            if tokens:
                tokenized_corpus.append(tokens)
                self._corpus_meta.append({
                    "element_id": rec.element_id,
                    "page": rec.page,
                    "modality": rec.modality,
                    "document_id": rec.document_id,
                    "content": rec.text,
                })

        for rec in table_records:
            tokens = _tokenize(rec.text)
            if tokens:
                tokenized_corpus.append(tokens)
                self._corpus_meta.append({
                    "element_id": rec.element_id,
                    "page": rec.page,
                    "modality": rec.modality,
                    "document_id": rec.document_id,
                    "content": rec.text,
                })

        if not tokenized_corpus:
            raise ValueError("Corpus vide : aucun token généré. Vérifiez les enregistrements textuels.")

        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25] Index construit : {len(tokenized_corpus)} documents tokenisés.")

    def search(self, query: str, top_k: int = 20) -> List[SearchHit]:
        """Recherche BM25 (lexicale) : retourne les top_k passages les plus pertinents."""
        if self._bm25 is None:
            raise RuntimeError("BM25Index non initialisé. Appelez build() d'abord.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(-scores)[:top_k]

        hits: List[SearchHit] = []
        for rank, idx in enumerate(top_indices, 1):
            meta = self._corpus_meta[idx]
            if scores[idx] <= 0.0:
                break  # BM25 score = 0 → aucune correspondance lexicale réelle
            hits.append(
                SearchHit(
                    element_id=meta["element_id"],
                    page=meta["page"],
                    modality=meta["modality"],
                    score=float(scores[idx]),
                    rank=rank,
                    metadata={
                        "document_id": meta["document_id"],
                        "content": meta["content"],
                        "source": "bm25",
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────────────────
# Fusion RRF générique (sur SearchHit, par element_id — plus précis que par page)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_hits_rrf(
    hit_lists: List[Tuple[str, List[SearchHit]]],
    k: int = 60,
    top_n: int = 20,
) -> List[SearchHit]:
    """Fusionne plusieurs listes de SearchHit par Reciprocal Rank Fusion (RRF).

    Args:
        hit_lists: liste de (nom_source, [SearchHit]) — ex: [("dense", hits1), ("bm25", hits2)]
        k: constante de lissage RRF (défaut 60, valeur canonique de la littérature)
        top_n: nombre de résultats fusionnés à retourner

    Returns:
        Liste de SearchHit fusionnés, triés par score RRF décroissant.

    Formule mathématique :
        RRF(d) = Σ  1 / (k + rank_i(d))    pour chaque liste i contenant d

    Points d'attention :
      - Seuls les RANGS sont utilisés, jamais les scores bruts : un score BM25 (non borné,
        ex. 8.4) et un cosinus (dans [-1, 1]) ne sont pas comparables entre eux.
      - Si une même liste contient un élément plusieurs fois (ex. points Qdrant
        dupliqués), seul son meilleur rang compte : un doublon ne doit pas gonfler le score.
      - La provenance est conservée : ``retrieval_sources`` et ``source_ranks``
        (ex. {"dense": 3, "bm25": 1}) pour l'explicabilité et l'analyse d'erreurs.
    """
    # eid → {source → meilleur rang dans cette source}
    ranks: Dict[str, Dict[str, int]] = {}
    # eid → premier hit rencontré (représentant ; contenu identique quelle que soit la source)
    representative: Dict[str, SearchHit] = {}

    for source_name, hits in hit_lists:
        for hit in hits:
            eid = hit.element_id
            per_source = ranks.setdefault(eid, {})
            if source_name not in per_source or hit.rank < per_source[source_name]:
                per_source[source_name] = hit.rank
            representative.setdefault(eid, hit)

    scores = {
        eid: sum(1.0 / (k + r) for r in per_source.values())
        for eid, per_source in ranks.items()
    }
    # Tri stable par score décroissant : à égalité, l'ordre de première apparition est conservé
    sorted_eids = sorted(scores, key=lambda eid: -scores[eid])[:top_n]

    result: List[SearchHit] = []
    for new_rank, eid in enumerate(sorted_eids, 1):
        base_hit = representative[eid]
        metadata = {k_: v for k_, v in base_hit.metadata.items() if k_ != "source"}
        result.append(
            SearchHit(
                element_id=base_hit.element_id,
                page=base_hit.page,
                modality=base_hit.modality,
                score=round(scores[eid], 6),  # score RRF consolidé
                rank=new_rank,
                metadata={
                    **metadata,
                    "rrf_score": round(scores[eid], 6),
                    "retrieval_sources": sorted(ranks[eid]),
                    "source_ranks": dict(ranks[eid]),
                },
            )
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HybridRetriever — Façade unifiée
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """Façade de recherche hybride couplant Qdrant (dense BGE-M3) + BM25 (sparse).

    Utilisation :
        retriever = HybridRetriever(qdrant_store, embeddings_mgr, bm25_index)
        text_hits, visual_hits = retriever.retrieve(
            question="Que signifie le voyant orange sur le SAN ?",
            visual_query_en="storage SAN orange warning indicator LEDs",
        )
    """

    def __init__(self, qdrant_store, embeddings_mgr, bm25_index: BM25Index):
        self.qdrant = qdrant_store
        self.emb = embeddings_mgr
        self.bm25 = bm25_index

    def retrieve(
        self,
        question: str,
        visual_query_en: Optional[str] = None,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        visual_top_k: int = 5,
        fused_top_n: int = 6,
        rrf_k: int = 60,
    ) -> Tuple[List[SearchHit], List[SearchHit]]:
        """Recherche hybride complète.

        Returns:
            (text_hits, visual_hits)
            - text_hits  : Top-N passages texte/tableau fusionnés par RRF (dense + BM25)
            - visual_hits: Top-K schémas techniques par similarité CLIP
        """
        # 1. Recherche dense dans Qdrant (BGE-M3)
        dense_hits = self.qdrant.search_text(question, self.emb, top_k=dense_top_k)

        # 2. Recherche BM25 (lexicale / exacte)
        bm25_hits = self.bm25.search(question, top_k=bm25_top_k)

        # 3. Fusion RRF des deux listes de hits textuels
        fused_text = fuse_hits_rrf(
            [("dense", dense_hits), ("bm25", bm25_hits)],
            k=rrf_k,
            top_n=fused_top_n,
        )

        # 4. Recherche visuelle CLIP dans Qdrant (indépendante)
        visual_hits: List[SearchHit] = []
        if visual_query_en:
            visual_hits = self.qdrant.search_visual(visual_query_en, self.emb, top_k=visual_top_k)

        return fused_text, visual_hits
