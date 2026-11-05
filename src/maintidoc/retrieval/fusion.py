"""Module de fusion par Reciprocal Rank Fusion (RRF)."""
from typing import List, Dict, Any
from maintidoc.schemas import FusionResult


def fuse_rrf(text_hits: List[Dict[str, Any]], visual_hits: List[Dict[str, Any]], k: int = 60) -> List[FusionResult]:
    """Fusionne les classements texte et visuel au niveau des pages.
    
    Formule RRF : score(d) = sum( 1 / (k + rank_i(d)) )
    """
    totals: Dict[int, float] = {}
    contributors: Dict[int, List[str]] = {}

    for source_name, hits in [("texte", text_hits), ("visuel", visual_hits)]:
        pages_seen = set()
        for h in hits:
            page = h["page"]
            if page in pages_seen:
                continue
            pages_seen.add(page)
            rank = h.get("rank", 1)
            totals[page] = totals.get(page, 0.0) + (1.0 / (k + rank))
            contributors.setdefault(page, []).append(source_name)

    sorted_pages = sorted(totals.items(), key=lambda item: -item[1])
    
    results = [
        FusionResult(
            page=p,
            rrf_score=round(score, 6),
            sources=sorted(list(set(contributors.get(p, [])))),
        )
        for p, score in sorted_pages
    ]
    return results

