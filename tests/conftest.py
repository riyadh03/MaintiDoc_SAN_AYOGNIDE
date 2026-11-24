"""Fixtures partagées : fabriques de SearchHit sans dépendance à Qdrant ni aux modèles."""
from __future__ import annotations

import pytest

from maintidoc.schemas import SearchHit


@pytest.fixture
def make_hit():
    """Fabrique de SearchHit textuels. ``content`` par défaut : texte de ``n_chars`` caractères."""

    def _make(
        element_id: str,
        rank: int,
        page: int = 1,
        score: float = 0.5,
        content: str | None = None,
        n_chars: int = 40,
        modality: str = "text",
        **metadata,
    ) -> SearchHit:
        return SearchHit(
            element_id=element_id,
            page=page,
            modality=modality,
            score=score,
            rank=rank,
            metadata={
                "document_id": "SAN-MAN-001",
                "content": content if content is not None else "x" * n_chars,
                **metadata,
            },
        )

    return _make
