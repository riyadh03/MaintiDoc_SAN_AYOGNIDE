"""Extracteur modulaire de PDF (Textes, Tableaux et Schémas multimodaux)."""
from pathlib import Path
import hashlib
import io
import json
from typing import Tuple, List, Dict, Any
import pymupdf
import pandas as pd
from PIL import Image

from maintidoc.config import (
    DEFAULT_PDF_PATH,
    DOC_ID,
    PROCESSED_DATA_DIR,
    IMAGES_DIR,
    TABLES_DIR,
    TEXTS_DIR,
)
from maintidoc.schemas import TextRecord, TableRecord, ImageRecord


class PDFExtractor:
    """Extraction industrielle et traçable du manuel SAN."""

    def __init__(self, pdf_path: Path = DEFAULT_PDF_PATH, output_dir: Path = PROCESSED_DATA_DIR):
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
            })
        doc.close()
        return inventory

    def extract_all(self) -> Tuple[List[TextRecord], List[TableRecord], List[ImageRecord]]:
        """Extrait l'ensemble des textes, tableaux et images avec métadonnées complètes."""
        doc = pymupdf.open(stream=self.doc_bytes, filetype="pdf")
        
        text_records: List[TextRecord] = []
        table_records: List[TableRecord] = []
        image_records: List[ImageRecord] = []

        for page_num, page in enumerate(doc, 1):
            # 1. Extraction du texte
            raw_text = page.get_text("text", sort=True).strip()
            if raw_text:
                txt_rec = TextRecord(
                    document_id=DOC_ID,
                    document_sha256=self.pdf_sha256,
                    page=page_num,
                    element_id=f"page_{page_num:02d}_text",
                    modality="text",
                    text=raw_text,
                )
                text_records.append(txt_rec)
                (self.texts_dir / f"page_{page_num:02d}.txt").write_text(raw_text, encoding="utf-8")

            # 2. Extraction des tableaux
            for k, table in enumerate(page.find_tables().tables, 1):
                rows = table.extract()
                if not rows:
                    continue
                width = max(len(row or []) for row in rows)
                clean_rows = [[("" if c is None else str(c)) for c in (row or [])] + [""] * (width - len(row or [])) for row in rows]
                headers = [f"col_{i+1}" for i in range(width)]
                frame = pd.DataFrame(clean_rows, columns=headers)
                eid = f"p{page_num:02d}_table_{k:02d}"
                
                frame.to_csv(self.tables_dir / f"{eid}.csv", index=False)
                try:
                    markdown_table = frame.to_markdown(index=False)
                except Exception:
                    # Fallback robuste sans dépendance tierce tabulate
                    header_line = " | ".join(headers)
                    sep_line = " | ".join(["---"] * len(headers))
                    body_lines = [" | ".join(r) for r in clean_rows]
                    markdown_table = f"| {header_line} |\n| {sep_line} |\n" + "\n".join(f"| {b} |" for b in body_lines)
                (self.tables_dir / f"{eid}.md").write_text(markdown_table, encoding="utf-8")
                
                tab_rec = TableRecord(
                    document_id=DOC_ID,
                    document_sha256=self.pdf_sha256,
                    page=page_num,
                    element_id=eid,
                    modality="table",
                    bbox=list(table.bbox),
                    rows=clean_rows,
                    text=markdown_table,
                )
                table_records.append(tab_rec)

            # 3. Extraction des images et schémas
            for k, img in enumerate(page.get_images(full=True), 1):
                xref = img[0]
                payload = doc.extract_image(xref)
                if not payload:
                    continue
                eid = f"p{page_num:02d}_image_{k:02d}"
                ext = payload.get("ext", "png")
                img_path = self.images_dir / f"{eid}.{ext}"
                img_path.write_bytes(payload["image"])
                
                # Règle métier SAN : Les figures FIG-01 à FIG-05 se trouvent aux pages 2 à 6
                kind = "schema" if 2 <= page_num <= 6 else "illustration"
                figure_id = f"FIG-{page_num-1:02d}" if kind == "schema" else None
                rects = [list(r) for r in page.get_image_rects(xref)]
                
                img_rec = ImageRecord(
                    document_id=DOC_ID,
                    document_sha256=self.pdf_sha256,
                    page=page_num,
                    element_id=eid,
                    figure_id=figure_id,
                    modality=kind,
                    image_path=str(img_path),
                    image_sha256=hashlib.sha256(payload["image"]).hexdigest(),
                    rectangles=rects,
                    xref=xref,
                )
                image_records.append(img_rec)

        doc.close()

        # Sauvegarde des traces JSON consolidées
        (self.output_dir / "texts.json").write_text(
            json.dumps([r.model_dump() for r in text_records], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "tables.json").write_text(
            json.dumps([r.model_dump() for r in table_records], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "images.json").write_text(
            json.dumps([r.model_dump() for r in image_records], ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return text_records, table_records, image_records

