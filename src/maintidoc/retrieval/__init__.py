"""Module de recherche et fusion — Dense, BM25, RRF, Hybride."""
from .fusion import fuse_rrf
from .hybrid_retriever import BM25Index, HybridRetriever, fuse_hits_rrf

__all__ = ["fuse_rrf", "BM25Index", "HybridRetriever", "fuse_hits_rrf"]

