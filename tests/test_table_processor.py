"""Reconstruction des tableaux depuis les mots du PDF (sans dépendance au manuel)."""
import pymupdf
import pytest

from maintidoc.ingestion.table_processor import (
    clean_text,
    rebuild_rows,
    split_header,
    to_markdown,
    words_in_rect,
)


def _word(x0, y0, x1, y1, text, block=0, line=0, no=0):
    return (x0, y0, x1, y1, text, block, line, no)


def test_clean_text_repairs_broken_apostrophes_between_letters_only():
    assert clean_text("D·UN SYSTEME") == "D'UN SYSTEME"
    assert clean_text("controle d�acces") == "controle d'acces"
    # Point médian séparateur (entouré d'espaces) : conservé
    assert clean_text("SAN · FC") == "SAN · FC"


def test_words_in_rect_uses_word_centre_and_reading_order():
    words = [
        _word(10, 10, 40, 20, "Chemin", line=0, no=0),
        _word(42, 10, 52, 20, "FC", line=0, no=1),
        _word(10, 22, 50, 32, "degrade", line=1, no=0),
        _word(200, 10, 230, 20, "Ailleurs"),
    ]
    # Le mot « degrade » déborde légèrement en bas du rectangle : son centre (y=27) est dedans
    assert words_in_rect(words, (0, 0, 100, 30)) == "Chemin FC degrade"


def test_rebuild_rows_keeps_codes_intact_and_drops_empty_rows():
    words = [
        _word(10, 10, 40, 20, "Code"), _word(60, 10, 90, 20, "Etat", no=1),
        _word(10, 30, 40, 40, "AL-01", line=1), _word(60, 30, 90, 40, "Orange", line=1, no=1),
    ]
    cells = [
        [(0, 0, 50, 25), (50, 0, 100, 25)],
        [(0, 25, 50, 45), (50, 25, 100, 45)],
        [(0, 45, 50, 60), None],  # ligne vide + cellule fusionnée
    ]
    assert rebuild_rows(cells, words) == [["Code", "Etat"], ["AL-01", "Orange"]]


def test_split_header_detected_and_deduplicated():
    header, body = split_header([["Code", "", "Code"], ["AL-01", "x", "y"]], header_detected=True)
    assert header == ["Code", "col_2", "Code_2"]
    assert body == [["AL-01", "x", "y"]]


def test_split_header_not_detected_keeps_all_rows():
    header, body = split_header([["a", "b"]], header_detected=False)
    assert header == ["col_1", "col_2"] and body == [["a", "b"]]


def test_markdown_escapes_pipes():
    md = to_markdown(["A", "B"], [["x | y", "z"]])
    assert md.splitlines() == ["| A | B |", "| --- | --- |", "| x \\| y | z |"]


@pytest.fixture
def table_pdf(tmp_path):
    """PDF minimal avec un vrai tableau à bordures, comme dans le manuel SAN."""
    doc = pymupdf.open()
    page = doc.new_page()
    rows = [["Code", "Signification pour le simulateur"], ["AL-01", "Chemin FC degrade"], ["AL-03", "Volume indisponible"]]
    x = [50, 150, 450]
    y0, h = 100, 25
    for r, row in enumerate(rows):
        for c, txt in enumerate(row):
            rect = pymupdf.Rect(x[c], y0 + r * h, x[c + 1], y0 + (r + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y0 + 16), txt, fontsize=10)
    path = tmp_path / "table.pdf"
    doc.save(path)
    doc.close()
    return path


def test_extractor_tables_end_to_end(table_pdf, tmp_path):
    from maintidoc.ingestion.pdf_extractor import PDFExtractor

    _, tables, _ = PDFExtractor(table_pdf, tmp_path / "out", ocr_engine=None).extract_all()
    assert len(tables) == 1
    t = tables[0]
    assert t.header == ["Code", "Signification pour le simulateur"]
    assert t.rows == [["AL-01", "Chemin FC degrade"], ["AL-03", "Volume indisponible"]]
    assert "| AL-01 | Chemin FC degrade |" in t.text
