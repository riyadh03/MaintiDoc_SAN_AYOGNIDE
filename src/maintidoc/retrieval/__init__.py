"""Module de recherche — Dense, BM25, RRF, Reranking, sélection des preuves."""
from .fusion import fuse_rrf
from .hybrid_retriever import BM25Index, HybridRetriever, fuse_hits_rrf
from .reranker import CrossEncoderReranker
from .evidence_packer import EvidencePacker

__all__ = [
    "fuse_rrf",
    "BM25Index",
    "HybridRetriever",
    "fuse_hits_rrf",
    "CrossEncoderReranker",
    "EvidencePacker",
]
