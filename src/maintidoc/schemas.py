"""Modèles de données typés (Pydantic) pour MaintiDoc."""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


def make_citation_id(document_id: str, page: int, element_id: str) -> str:
    """Identifiant de citation canonique : ``<document_id>:p<page>:<element_id>``.

    Source unique de vérité du format : l'EvidencePacker l'utilise pour construire
    les citations autorisées, le prompt LLM pour étiqueter les sources, et
    ``validate_citations`` pour vérifier la réponse. Ne jamais reconstruire ce
    format à la main ailleurs.
    """
    return f"{document_id}:p{page}:{element_id}"


class BaseEvidence(BaseModel):
    """Preuve documentaire de base."""
    document_id: str = "SAN-MAN-001"
    document_sha256: str
    page: int
    element_id: str
    modality: str  # "text", "table", "schema_text", "schema", "illustration"


class TextRecord(BaseEvidence):
    """Segment textuel extrait d'une page.

    ``source`` trace l'origine du texte (provenance, explicabilité) :
      - ``pdf_text_layer`` : couche texte native du PDF (fiable) ;
      - ``ocr_page``       : page scannée reconnue par OCR (modality ``text``) ;
      - ``ocr_figure``     : étiquettes lues dans un schéma (modality ``schema_text``),
                             rattachées à l'image par ``linked_element_id``.
    """
    text: str
    source: str = "pdf_text_layer"
    ocr_confidence: Optional[float] = None  # confiance moyenne des lignes OCR retenues
    figure_id: Optional[str] = None
    linked_element_id: Optional[str] = None  # element_id de l'ImageRecord source


class TableRecord(BaseEvidence):
    """Tableau extrait d'une page avec métadonnées structurelles."""
    bbox: List[float]
    header: List[str] = Field(default_factory=list)
    rows: List[List[str]]  # corps du tableau (sans l'en-tête)
    text: str  # Format Markdown


class ImageRecord(BaseEvidence):
    """Image ou schéma extrait avec coordonnées."""
    figure_id: Optional[str] = None  # e.g. "FIG-01"
    image_path: str
    image_sha256: str
    rectangles: List[List[float]] = Field(default_factory=list)
    xref: int
    ocr_text: Optional[str] = None  # étiquettes lues par OCR (None si OCR désactivé/vide)


class SearchHit(BaseModel):
    """Résultat unitaire d'une recherche vectorielle."""
    element_id: str
    page: int
    modality: str
    score: float
    rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FusionResult(BaseModel):
    """Résultat fusionné par RRF."""
    page: int
    rrf_score: float
    sources: List[str]  # e.g. ["texte", "visuel"]

