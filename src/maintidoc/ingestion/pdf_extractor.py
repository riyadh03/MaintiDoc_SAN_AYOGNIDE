"""Extracteur modulaire de PDF (Textes, Tableaux, Schémas multimodaux et OCR)."""
from pathlib import Path
import hashlib
import io
import json
import logging
from typing import Tuple, List, Dict, Any, Optional

import pymupdf
import pandas as pd
from PIL import Image

from maintidoc.config import (
    DEFAULT_PDF_PATH,
    DOC_ID,
    PROCESSED_DATA_DIR,
    OCR_ENABLED,
    OCR_PAGE_DPI,
    OCR_PAGE_MAX_TEXT_CHARS,
    OCR_PAGE_MIN_IMAGE_COVERAGE,
)
from maintidoc.ingestion.ocr import (
    OCREngine,
    default_engine,
    detect_figure_id,
    is_meaningful,
    lines_to_text,
    mean_confidence,
)
from maintidoc.ingestion.table_processor import clean_text, rebuild_rows, split_header, to_markdown
from maintidoc.schemas import TextRecord, TableRecord, ImageRecord

logger = logging.getLogger(__name__)

_DEFAULT = object()  # sentinelle : « moteur OCR par défaut » ≠ None (« OCR désactivé »)


class PDFExtractor:
    """Extraction industrielle et traçable du manuel SAN.

    Args:
        ocr_engine: moteur OCR à utiliser. Par défaut RapidOCR (si installé et
            ``OCR_ENABLED``) ; ``None`` désactive explicitement l'OCR.
    """

    def __init__(
        self,
        pdf_path: Path = DEFAULT_PDF_PATH,
        output_dir: Path = PROCESSED_DATA_DIR,
        ocr_engine: Optional[OCREngine] = _DEFAULT,  # type: ignore[assignment]
    ):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {self.pdf_path}")

        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        self.texts_dir = self.output_dir / "page_texts"

        # Création des répertoires de sortie
        for d in [self.output_dir, self.images_dir, self.tables_dir, self.texts_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.doc_bytes = self.pdf_path.read_bytes()
        self.pdf_sha256 = hashlib.sha256(self.doc_bytes).hexdigest()

        if ocr_engine is _DEFAULT:
            ocr_engine = default_engine() if OCR_ENABLED else None
        self.ocr: Optional[OCREngine] = ocr_engine

    def get_inventory(self) -> List[Dict[str, Any]]:
        """Retourne l'inventaire rapide de chaque page du PDF."""
        doc = pymupdf.open(stream=self.doc_bytes, filetype="pdf")
        inventory = []
        for page_num, page in enumerate(doc, 1):
            inventory.append({
                "page": page_num,
                "text_chars": len(page.get_text("text")),
                "embedded_images": len(page.get_images(full=True)),
                "detected_tables": len(page.find_tables().tables),
                "image_coverage": round(self._image_coverage(page), 3),
            })
        doc.close()
        return inventory

    def extract_all(self) -> Tuple[List[TextRecord], List[TableRecord], List[ImageRecord]]:
        """Extrait textes, tableaux et images avec métadonnées complètes.

        Les étiquettes OCR des schémas sont renvoyées dans la liste des *textes*
        (modality ``schema_text``) : elles rejoignent ainsi automatiquement l'index
        dense Qdrant et l'index BM25, sans changer le contrat de l'appelant.
        """
        doc = pymupdf.open(stream=self.doc_bytes, filetype="pdf")

        text_records: List[TextRecord] = []
        table_records: List[TableRecord] = []
        image_records: List[ImageRecord] = []

        for page_num, page in enumerate(doc, 1):
            page_text = self._extract_page_text(page, page_num)
            if page_text:
                text_records.append(page_text)

            table_records.extend(self._extract_tables(page, page_num))

            images, ocr_texts = self._extract_images(doc, page, page_num)
            image_records.extend(images)
            text_records.extend(ocr_texts)

        doc.close()

        # Sauvegarde des traces JSON consolidées
        for name, records in (("texts", text_records), ("tables", table_records), ("images", image_records)):
            (self.output_dir / f"{name}.json").write_text(
                json.dumps([r.model_dump() for r in records], ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return text_records, table_records, image_records

    # ─────────────────────────────────────────────────────────────────────
    # 1. Texte de page (couche native, ou OCR si page scannée)
    # ─────────────────────────────────────────────────────────────────────

    def _extract_page_text(self, page: "pymupdf.Page", page_num: int) -> Optional[TextRecord]:
        raw_text = clean_text_block(page.get_text("text", sort=True))
        source, confidence = "pdf_text_layer", None

        if self.ocr is not None and self._is_scanned(page, raw_text):
            ocr_text, confidence = self._ocr_page(page)
            if ocr_text:
                # On conserve la couche native éventuelle (en-tête/pied) devant le texte OCR
                raw_text = f"{raw_text}\n{ocr_text}".strip() if raw_text else ocr_text
                source = "ocr_page"
                logger.info("Page %d : couche texte absente, %d caractères reconnus par OCR.", page_num, len(ocr_text))

        if not raw_text:
            return None
        (self.texts_dir / f"page_{page_num:02d}.txt").write_text(raw_text, encoding="utf-8")
        return TextRecord(
            document_id=DOC_ID,
            document_sha256=self.pdf_sha256,
            page=page_num,
            element_id=f"page_{page_num:02d}_text",
            modality="text",
            text=raw_text,
            source=source,
            ocr_confidence=confidence,
        )

    @staticmethod
    def _image_coverage(page: "pymupdf.Page") -> float:
        """Part de la page couverte par sa plus grande image (0 → 1)."""
        page_area = page.rect.get_area() or 1.0
        best = 0.0
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]):
                best = max(best, (r & page.rect).get_area() / page_area)
        return best

    def _is_scanned(self, page: "pymupdf.Page", native_text: str) -> bool:
        return (
            len(native_text) < OCR_PAGE_MAX_TEXT_CHARS
            and self._image_coverage(page) >= OCR_PAGE_MIN_IMAGE_COVERAGE
        )

    def _ocr_page(self, page: "pymupdf.Page") -> Tuple[str, Optional[float]]:
        pix = page.get_pixmap(dpi=OCR_PAGE_DPI)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        lines = self.ocr.recognize(image)
        text = lines_to_text(lines)
        if not is_meaningful(text):
            return "", None
        return text, mean_confidence(lines)

    # ─────────────────────────────────────────────────────────────────────
    # 2. Tableaux (géométrie find_tables + contenu reconstruit depuis les mots)
    # ─────────────────────────────────────────────────────────────────────

    def _extract_tables(self, page: "pymupdf.Page", page_num: int) -> List[TableRecord]:
        words = page.get_text("words")
        records: List[TableRecord] = []
        for k, table in enumerate(page.find_tables().tables, 1):
            rows = rebuild_rows([row.cells for row in table.rows], words)
            if not rows:
                continue
            header, body = split_header(rows, header_detected=not table.header.external)
            eid = f"p{page_num:02d}_table_{k:02d}"
            markdown_table = to_markdown(header, body)

            pd.DataFrame(body, columns=header).to_csv(self.tables_dir / f"{eid}.csv", index=False)
            (self.tables_dir / f"{eid}.md").write_text(markdown_table, encoding="utf-8")

            records.append(TableRecord(
                document_id=DOC_ID,
                document_sha256=self.pdf_sha256,
                page=page_num,
                element_id=eid,
                modality="table",
                bbox=list(table.bbox),
                header=header,
                rows=body,
                text=markdown_table,
            ))
        return records

    # ─────────────────────────────────────────────────────────────────────
    # 3. Images et schémas (+ OCR des étiquettes)
    # ─────────────────────────────────────────────────────────────────────

    def _extract_images(
        self, doc: "pymupdf.Document", page: "pymupdf.Page", page_num: int
    ) -> Tuple[List[ImageRecord], List[TextRecord]]:
        images: List[ImageRecord] = []
        ocr_texts: List[TextRecord] = []

        for k, img in enumerate(page.get_images(full=True), 1):
            xref = img[0]
            payload = doc.extract_image(xref)
            if not payload:
                continue
            eid = f"p{page_num:02d}_image_{k:02d}"
            ext = payload.get("ext", "png")
            img_path = self.images_dir / f"{eid}.{ext}"
            img_path.write_bytes(payload["image"])

            ocr_text, confidence = "", None
            if self.ocr is not None:
                try:
                    lines = self.ocr.recognize(Image.open(io.BytesIO(payload["image"])))
                    ocr_text, confidence = lines_to_text(lines), mean_confidence(lines)
                    if not is_meaningful(ocr_text):
                        ocr_text, confidence = "", None
                except Exception:  # une image illisible ne doit pas interrompre l'extraction
                    logger.exception("OCR échoué sur %s", eid)

            # Identifiant de figure : d'abord celui LU dans le titre du schéma (preuve directe),
            # sinon la règle métier SAN (FIG-01 à FIG-05 aux pages 2 à 6).
            figure_id = detect_figure_id(ocr_text)
            if figure_id is None and 2 <= page_num <= 6:
                figure_id = f"FIG-{page_num - 1:02d}"
            kind = "schema" if figure_id else "illustration"

            images.append(ImageRecord(
                document_id=DOC_ID,
                document_sha256=self.pdf_sha256,
                page=page_num,
                element_id=eid,
                figure_id=figure_id,
                modality=kind,
                image_path=str(img_path),
                image_sha256=hashlib.sha256(payload["image"]).hexdigest(),
                rectangles=[list(r) for r in page.get_image_rects(xref)],
                xref=xref,
                ocr_text=ocr_text or None,
            ))

            if ocr_text:
                ocr_texts.append(TextRecord(
                    document_id=DOC_ID,
                    document_sha256=self.pdf_sha256,
                    page=page_num,
                    element_id=f"{eid}_ocr",
                    modality="schema_text" if kind == "schema" else "image_text",
                    text=ocr_text,
                    source="ocr_figure",
                    ocr_confidence=confidence,
                    figure_id=figure_id,
                    linked_element_id=eid,
                ))

        return images, ocr_texts


def clean_text_block(text: str) -> str:
    """``clean_text`` ligne par ligne (conserve les retours à la ligne de la page)."""
    return "\n".join(filter(None, (clean_text(line) for line in text.splitlines())))
