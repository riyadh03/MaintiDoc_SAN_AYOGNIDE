"""Gestionnaire de base de données vectorielle Qdrant pour MaintiDoc."""
from typing import List, Dict, Any, Optional
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from maintidoc.schemas import TextRecord, TableRecord, ImageRecord, SearchHit
from maintidoc.indexing.embeddings import EmbeddingsManager
from maintidoc.config import INDICES_DIR


class QdrantStore:
    """Gestionnaire de collections vectorielles Qdrant."""

    TEXT_COLLECTION = "maintidoc_text"
    VISUAL_COLLECTION = "maintidoc_visual"
    TEXT_DIM = 1024  # BGE-M3
    VISUAL_DIM = 512  # CLIP ViT-B/32

    def __init__(self, host: Optional[str] = None, port: int = 6333, path: Optional[str] = None):
        # 1. Tentative de connexion au serveur Qdrant (Docker ou distant)
        target_host = host or os.getenv("QDRANT_HOST", "localhost")
        self.mode = "remote"
        try:
            self.client = QdrantClient(host=target_host, port=port, timeout=2.0)
            self.client.get_collections()
            self.is_connected = True
        except Exception:
            # 2. Fallback persistant sur disque local (sans dépendance externe Docker)
            db_path = path or str(INDICES_DIR / "qdrant_db")
            os.makedirs(db_path, exist_ok=True)
            self.client = QdrantClient(path=db_path)
            self.is_connected = False
            self.mode = f"local_disk ({db_path})"

    def init_collections(self, force_recreate: bool = False):
        """Initialise les deux collections (texte et visuel) avec leurs dimensions respectives."""
        collections = [c.name for c in self.client.get_collections().collections]

        if force_recreate or self.TEXT_COLLECTION not in collections:
            if self.TEXT_COLLECTION in collections:
                self.client.delete_collection(self.TEXT_COLLECTION)
            self.client.create_collection(
                collection_name=self.TEXT_COLLECTION,
                vectors_config=VectorParams(size=self.TEXT_DIM, distance=Distance.COSINE),
            )

        if force_recreate or self.VISUAL_COLLECTION not in collections:
            if self.VISUAL_COLLECTION in collections:
                self.client.delete_collection(self.VISUAL_COLLECTION)
            self.client.create_collection(
                collection_name=self.VISUAL_COLLECTION,
                vectors_config=VectorParams(size=self.VISUAL_DIM, distance=Distance.COSINE),
            )

    def index_texts_and_tables(self, texts: List[TextRecord], tables: List[TableRecord], emb_mgr: EmbeddingsManager):
        """Indexe les paragraphes de texte et les tableaux Markdown."""
        points: List[PointStruct] = []
        
        # 1. Traitement des textes
        for t in texts:
            vector = emb_mgr.encode_text(t.text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "document_id": t.document_id,
                        "document_sha256": t.document_sha256,
                        "page": t.page,
                        "element_id": t.element_id,
                        "modality": t.modality,
                        "content": t.text,
                    },
                )
            )

        # 2. Traitement des tableaux
        for tab in tables:
            vector = emb_mgr.encode_text(tab.text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "document_id": tab.document_id,
                        "document_sha256": tab.document_sha256,
                        "page": tab.page,
                        "element_id": tab.element_id,
                        "modality": tab.modality,
                        "content": tab.text,
                        "bbox": tab.bbox,
                    },
                )
            )

        if points:
            self.client.upsert(collection_name=self.TEXT_COLLECTION, points=points)

    def index_images(self, images: List[ImageRecord], emb_mgr: EmbeddingsManager):
        """Indexe les schémas et images techniques avec CLIP."""
        points: List[PointStruct] = []
        for im in images:
            vector = emb_mgr.encode_image(im.image_path)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "document_id": im.document_id,
                        "document_sha256": im.document_sha256,
                        "page": im.page,
                        "element_id": im.element_id,
                        "figure_id": im.figure_id,
                        "modality": im.modality,
                        "image_path": im.image_path,
                        "image_sha256": im.image_sha256,
                        "rectangles": im.rectangles,
                    },
                )
            )

        if points:
            self.client.upsert(collection_name=self.VISUAL_COLLECTION, points=points)

    def search_text(self, query: str, emb_mgr: EmbeddingsManager, top_k: int = 6) -> List[SearchHit]:
        """Recherche sémantique dense dans les textes et tableaux."""
        query_vector = emb_mgr.encode_text(query)
        hits = self.client.search(
            collection_name=self.TEXT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            SearchHit(
                element_id=h.payload.get("element_id", ""),
                page=int(h.payload.get("page", 0)),
                modality=h.payload.get("modality", "text"),
                score=float(h.score),
                rank=i,
                metadata={
                    "document_id": h.payload.get("document_id"),
                    "content": h.payload.get("content", ""),
                },
            )
            for i, h in enumerate(hits, 1)
        ]

    def search_visual(self, visual_query_en: str, emb_mgr: EmbeddingsManager, top_k: int = 5) -> List[SearchHit]:
        """Recherche multimodale dans les schémas via CLIP."""
        query_vector = emb_mgr.encode_visual_query(visual_query_en)
        hits = self.client.search(
            collection_name=self.VISUAL_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            SearchHit(
                element_id=h.payload.get("element_id", ""),
                page=int(h.payload.get("page", 0)),
                modality=h.payload.get("modality", "schema"),
                score=float(h.score),
                rank=i,
                metadata={
                    "document_id": h.payload.get("document_id"),
                    "figure_id": h.payload.get("figure_id"),
                    "image_path": h.payload.get("image_path"),
                },
            )
            for i, h in enumerate(hits, 1)
        ]

