"""OCR : post-traitement, ordre de lecture, et intégration dans l'extracteur (moteur factice)."""
import io

import pymupdf
import pytest
from PIL import Image, ImageDraw

from maintidoc.config import DEFAULT_PDF_PATH
from maintidoc.ingestion.ocr import (
    OCRLine,
    detect_figure_id,
    is_meaningful,
    lines_to_text,
    normalize_ocr_text,
)
from maintidoc.ingestion.pdf_extractor import PDFExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Post-correction (erreurs réellement observées sur les schémas du manuel)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("FlG-04|Voyants de statut", "FIG-04 | Voyants de statut"),
    ("FiG-03 | Controleurs", "FIG-03 | Controleurs"),
    ("FlG-o5IWorkflow", "FIG-05 | Workflow"),
    ("FIG-02 | Fabric Fibre Channel", "FIG-02 | Fabric Fibre Channel"),
    ("Verifier zoning, masking, MPlO et etat", "Verifier zoning, masking, MPIO et etat"),
])
def test_normalize_fixes_code_confusions(raw, expected):
    assert normalize_ocr_text(raw) == expected


def test_normalize_leaves_ordinary_words_alone():
    text = "Il faut lire le flux de la baie"
    assert normalize_ocr_text(text) == text


def test_detect_figure_id():
    assert detect_figure_id("FIG-02 | Fabric\nHBA 1") == "FIG-02"
    assert detect_figure_id("Panneau de supervision") is None


def _line(text, x, y, conf=0.99, h=10):
    return OCRLine(text=text, confidence=conf, bbox=(x, y, x + 50, y + h))


def test_lines_to_text_reading_order_and_confidence_filter():
    lines = [
        _line("Switch FC A", 200, 101),  # même rangée que HBA 1 (décalage de 1 px)
        _line("HBA 1", 10, 100),
        _line("WWPN H1", 10, 130),
        _line("~~", 300, 130, conf=0.2),  # bruit sous le seuil
    ]
    assert lines_to_text(lines) == "HBA 1 ; Switch FC A\nWWPN H1"


def test_is_meaningful_rejects_isolated_letters():
    assert not is_meaningful("A")
    assert is_meaningful("HBA 1")


# ─────────────────────────────────────────────────────────────────────────────
# Intégration extracteur avec moteur factice (aucun modèle chargé)
# ─────────────────────────────────────────────────────────────────────────────

class FakeOCR:
    """Renvoie un texte fixe et compte les appels."""

    def __init__(self, text="FlG-02|Fabric Fibre Channel"):
        self.text = text
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return [OCRLine(self.text, 0.97, (0, 0, 100, 10)), OCRLine("HBA 1", 0.95, (0, 40, 40, 50))]


def _png(w=400, h=200):
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).rectangle((10, 10, w - 10, h - 10), outline="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf(tmp_path, build):
    doc = pymupdf.open()
    build(doc)
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


def test_figure_ocr_becomes_searchable_text_record(tmp_path):
    def build(doc):
        page = doc.new_page()
        page.insert_text((50, 50), "Texte natif de la page avec suffisamment de contenu pour ne pas etre scannee.")
        page.insert_image(pymupdf.Rect(50, 100, 450, 300), stream=_png())

    ocr = FakeOCR()
    texts, _, images = PDFExtractor(_pdf(tmp_path, build), tmp_path / "out", ocr_engine=ocr).extract_all()

    assert ocr.calls == 1  # l'image seulement : la page a une couche texte
    assert images[0].figure_id == "FIG-02"  # lu dans le titre, pas déduit de la page
    assert images[0].modality == "schema"
    ocr_rec = [t for t in texts if t.source == "ocr_figure"]
    assert len(ocr_rec) == 1
    rec = ocr_rec[0]
    assert rec.modality == "schema_text"
    assert rec.linked_element_id == images[0].element_id
    assert rec.element_id == f"{images[0].element_id}_ocr"
    assert rec.text == "FIG-02 | Fabric Fibre Channel\nHBA 1"
    assert rec.ocr_confidence == pytest.approx(0.96)


def test_scanned_page_is_ocred(tmp_path):
    def build(doc):
        page = doc.new_page()
        page.insert_image(page.rect, stream=_png(600, 800))  # page = une seule image pleine page

    ocr = FakeOCR(text="Procedure de remplacement du disque B03")
    texts, _, _ = PDFExtractor(_pdf(tmp_path, build), tmp_path / "out", ocr_engine=ocr).extract_all()

    page_rec = [t for t in texts if t.element_id == "page_01_text"]
    assert len(page_rec) == 1
    assert page_rec[0].source == "ocr_page"
    assert "Procedure de remplacement du disque B03" in page_rec[0].text


def test_blank_page_does_not_trigger_ocr(tmp_path):
    """Cas réel de la page 8 du manuel : pied de page seul, aucune image."""
    def build(doc):
        doc.new_page().insert_text((50, 800), "SAN-MAN-001 | Page 8")

    ocr = FakeOCR()
    texts, _, _ = PDFExtractor(_pdf(tmp_path, build), tmp_path / "out", ocr_engine=ocr).extract_all()
    assert ocr.calls == 0
    assert texts[0].source == "pdf_text_layer"


def test_ocr_disabled_keeps_extraction_working(tmp_path):
    def build(doc):
        page = doc.new_page()
        page.insert_text((50, 50), "Texte")
        page.insert_image(pymupdf.Rect(50, 100, 450, 300), stream=_png())

    texts, _, images = PDFExtractor(_pdf(tmp_path, build), tmp_path / "out", ocr_engine=None).extract_all()
    assert len(images) == 1 and images[0].ocr_text is None
    assert all(t.source == "pdf_text_layer" for t in texts)


# ─────────────────────────────────────────────────────────────────────────────
# Intégration réelle : manuel SAN + RapidOCR (lent, ignoré si indisponible)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_real_manual_diagrams_are_readable(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    if not DEFAULT_PDF_PATH.exists():
        pytest.skip("Manuel SAN absent")

    texts, tables, images = PDFExtractor(DEFAULT_PDF_PATH, tmp_path).extract_all()

    schemas = {i.figure_id for i in images if i.modality == "schema"}
    assert schemas == {"FIG-01", "FIG-02", "FIG-03", "FIG-04", "FIG-05"}

    ocr_by_fig = {t.figure_id: t.text for t in texts if t.source == "ocr_figure"}
    assert "WWPN H1" in ocr_by_fig["FIG-02"]
    assert "MPIO" in ocr_by_fig["FIG-02"]
    assert "Defaut critique" in ocr_by_fig["FIG-04"]

    alarm = next(t for t in tables if t.header and t.header[0] == "Code")
    assert alarm.rows[0][:2] == ["AL-01", "Orange"]
    assert "DIAG-FC-01" in alarm.rows[0][3]
    assert not any("·" in t.text for t in texts)
