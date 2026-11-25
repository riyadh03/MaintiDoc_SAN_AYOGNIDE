"""OCR des schémas techniques et des pages scannées.

Pourquoi ce module existe :
  - Les schémas FIG-01 à FIG-05 sont des images raster. CLIP les indexe *visuellement*
    mais ne sait pas LIRE leurs étiquettes (« HBA 1 », « WWPN H1 », « Defaut critique »).
    Sans OCR, ces termes — exactement ceux qu'un technicien tape — sont invisibles
    pour BM25 et pour la recherche dense.
  - Un manuel constructeur scanné n'a pas de couche texte : sans OCR, la page entière
    est perdue pour la recherche.

Choix du moteur : RapidOCR (modèles PaddleOCR exportés en ONNX).
  - Installation pip pure (pas de binaire système comme Tesseract), modèles embarqués
    dans la wheel → reproductible sous Windows comme dans le conteneur Docker.
  - CPU suffisant : ~4-5 s par schéma 1300x570 après agrandissement x2 (traitement hors ligne).
  - Le moteur est isolé derrière le protocole ``OCREngine`` : le remplacer (EasyOCR,
    Tesseract, VLM) ne touche pas l'extracteur.

Faiblesses connues du modèle, corrigées ici :
  - Espaces perdus sur le texte de petite taille → agrandissement x2 avant reconnaissance.
  - Confusions I/l/i et 0/o dans les codes (« FlG-o5 », « MPlO ») → post-correction
    ciblée sur les seuls jetons qui ont la forme d'un code technique.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from maintidoc.config import OCR_MIN_ALNUM_CHARS, OCR_MIN_CONFIDENCE, OCR_UPSCALE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRLine:
    """Ligne de texte reconnue. ``bbox`` = (x0, y0, x1, y1) en pixels de l'image d'origine."""
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


class OCREngine(Protocol):
    """Contrat minimal d'un moteur OCR : image RGB → lignes reconnues."""

    def recognize(self, image: Image.Image) -> List[OCRLine]: ...


class RapidOCREngine:
    """Moteur RapidOCR (ONNX Runtime), chargé paresseusement au premier appel."""

    def __init__(self, upscale: int = OCR_UPSCALE):
        if upscale < 1:
            raise ValueError("upscale doit être >= 1")
        self.upscale = upscale
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as e:
                raise ImportError("Installer l'OCR : uv add rapidocr-onnxruntime") from e
            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image: Image.Image) -> List[OCRLine]:
        engine = self._load()
        img = image.convert("RGB")
        if self.upscale > 1:
            img = img.resize((img.width * self.upscale, img.height * self.upscale), Image.LANCZOS)
        result, _ = engine(np.array(img))
        lines: List[OCRLine] = []
        for box, text, conf in result or []:
            xs = [p[0] / self.upscale for p in box]
            ys = [p[1] / self.upscale for p in box]
            lines.append(OCRLine(text=text, confidence=float(conf), bbox=(min(xs), min(ys), max(xs), max(ys))))
        return lines


def default_engine() -> Optional[OCREngine]:
    """Moteur par défaut, ou ``None`` (avec avertissement) si RapidOCR n'est pas installé.

    Dégradation contrôlée : sans OCR, l'extraction continue (textes, tableaux, images
    CLIP) ; seuls les enregistrements OCR sont absents.
    """
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        logger.warning("rapidocr-onnxruntime absent : OCR des schémas et pages scannées désactivé.")
        return None
    return RapidOCREngine()


# ─────────────────────────────────────────────────────────────────────────────
# Post-traitement
# ─────────────────────────────────────────────────────────────────────────────

# « FIG-05 » mal lu : F + (I|l|1|i) + (G|6) + séparateur optionnel + 2 « chiffres » (o/O/l/I tolérés)
_FIG_RE = re.compile(r"\bF[Il1i][Gg6]\s*[-_.]?\s*([0-9oOlI]{2})(?=[^0-9A-Za-z]|[A-Z|]|$)")
_DIGIT_FIX = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1"})
# Jeton majuscule contenant un « l » isolé entre majuscules : MPlO → MPIO
_CAPS_L_RE = re.compile(r"\b([A-Z]{1,})l([A-Z]{1,})\b")
# Barre verticale de titre (« FIG-02 | Titre ») souvent lue « I » ou collée
_TITLE_BAR_RE = re.compile(r"^(FIG-\d{2})\s*[|I]?\s*(?=\S)")

FIGURE_ID_RE = re.compile(r"\bFIG-\d{2}\b")


def normalize_ocr_text(text: str) -> str:
    """Corrige les confusions de caractères typiques sur les codes techniques.

    Volontairement étroit : on ne touche qu'aux jetons ayant la forme d'un code
    (identifiant de figure, acronyme en majuscules). Le texte courant n'est pas
    « corrigé » : une fausse correction serait pire qu'une erreur visible.
    """
    text = _FIG_RE.sub(lambda m: f"FIG-{m.group(1).translate(_DIGIT_FIX)}", text)
    text = _CAPS_L_RE.sub(lambda m: f"{m.group(1)}I{m.group(2)}", text)
    text = _TITLE_BAR_RE.sub(lambda m: f"{m.group(1)} | ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def lines_to_text(lines: Sequence[OCRLine], min_confidence: float = OCR_MIN_CONFIDENCE) -> str:
    """Reconstruit un texte en ordre de lecture (haut→bas, puis gauche→droite).

    Deux lignes appartiennent à la même rangée visuelle si leurs centres verticaux
    sont plus proches que la moitié de la hauteur médiane d'une ligne : l'OCR renvoie
    des boîtes légèrement décalées pour des étiquettes pourtant alignées.
    """
    kept = [ln for ln in lines if ln.confidence >= min_confidence and ln.text.strip()]
    if not kept:
        return ""
    heights = sorted(ln.bbox[3] - ln.bbox[1] for ln in kept)
    tolerance = max(heights[len(heights) // 2] / 2, 1.0)

    rows: List[List[OCRLine]] = []
    for ln in sorted(kept, key=lambda l_: (l_.bbox[1] + l_.bbox[3]) / 2):
        center = (ln.bbox[1] + ln.bbox[3]) / 2
        if rows:
            last = rows[-1]
            last_center = sum((l_.bbox[1] + l_.bbox[3]) / 2 for l_ in last) / len(last)
            if abs(center - last_center) <= tolerance:
                last.append(ln)
                continue
        rows.append([ln])

    out = []
    for row in rows:
        row.sort(key=lambda l_: l_.bbox[0])
        out.append(" ; ".join(normalize_ocr_text(l_.text) for l_ in row))
    return "\n".join(out)


def is_meaningful(text: str, min_alnum: int = OCR_MIN_ALNUM_CHARS) -> bool:
    """Écarte le bruit (une lettre isolée lue dans une photo de couverture, etc.)."""
    return sum(ch.isalnum() for ch in text) >= min_alnum


def mean_confidence(lines: Sequence[OCRLine], min_confidence: float = OCR_MIN_CONFIDENCE) -> Optional[float]:
    kept = [ln.confidence for ln in lines if ln.confidence >= min_confidence and ln.text.strip()]
    return round(sum(kept) / len(kept), 4) if kept else None


def detect_figure_id(text: str) -> Optional[str]:
    """Premier identifiant « FIG-NN » trouvé dans le texte OCR normalisé (le titre)."""
    m = FIGURE_ID_RE.search(text)
    return m.group(0) if m else None
