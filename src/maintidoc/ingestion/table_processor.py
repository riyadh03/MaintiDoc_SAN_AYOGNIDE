"""Reconstruction fidèle des tableaux PDF.

Problème constaté sur le manuel SAN avec ``Table.extract()`` (PyMuPDF 2.x) :
  - espaces perdus : « Symptomepossible », « Significationpourlesimulateur » ;
  - codes cassés en deux lignes : « AL-01 » devient « AL01 » + une ligne fantôme « - ».
    C'est rédhibitoire pour BM25, dont le rôle est précisément de retrouver les codes exacts ;
  - point médian « · » (U+00B7) utilisé comme apostrophe (« D·UN », « d·acces »).

Or la couche texte du PDF est correcte (``page.get_text("words")`` renvoie bien « AL-01 »,
« DIAG-FC-01 »). On garde donc la *géométrie* détectée par ``find_tables`` (rectangles des
cellules) et on reconstruit le *contenu* de chaque cellule à partir des mots dont le centre
tombe dans la cellule, dans l'ordre de lecture.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# Mot PyMuPDF : (x0, y0, x1, y1, texte, block_no, line_no, word_no)
Word = Tuple[float, float, float, float, str, int, int, int]

# Apostrophe mal encodée entre deux lettres : point médian (U+00B7, cas du manuel SAN)
# ou caractère de remplacement (U+FFFD). Entre deux lettres uniquement : un « · »
# séparateur entouré d'espaces reste intact.
_MIDDLE_DOT, _REPLACEMENT_CHAR = chr(0x00B7), chr(0xFFFD)
_BROKEN_APOSTROPHE_RE = re.compile(rf"(?<=\w)[{_MIDDLE_DOT}{_REPLACEMENT_CHAR}](?=\w)")


def clean_text(text: str) -> str:
    """Normalisation commune texte/tableaux : apostrophes cassées, espaces multiples."""
    text = _BROKEN_APOSTROPHE_RE.sub("'", text)
    text = text.replace(_REPLACEMENT_CHAR, "")
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def words_in_rect(words: Sequence[Word], rect: Sequence[float]) -> str:
    """Concatène les mots dont le centre est dans ``rect``, en ordre de lecture.

    Le critère du *centre* (et non de l'inclusion stricte) tolère les débordements
    d'un ou deux points des glyphes sur les bordures de cellule.
    """
    x0, y0, x1, y1 = rect
    inside = [
        w for w in words
        if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1
    ]
    inside.sort(key=lambda w: (w[5], w[6], w[7]))  # bloc, ligne, mot : ordre natif du PDF
    return clean_text(" ".join(w[4] for w in inside))


def rebuild_rows(cell_rows: Sequence[Sequence[Optional[Sequence[float]]]], words: Sequence[Word]) -> List[List[str]]:
    """Reconstruit la grille de texte à partir des rectangles de cellules.

    ``cell_rows`` : pour chaque ligne, la liste des rectangles de cellules
    (``None`` pour une cellule fusionnée/absente → chaîne vide).
    Les lignes entièrement vides sont supprimées.
    """
    width = max((len(r) for r in cell_rows), default=0)
    rows: List[List[str]] = []
    for cells in cell_rows:
        row = [words_in_rect(words, c) if c is not None else "" for c in cells]
        row += [""] * (width - len(row))
        if any(row):
            rows.append(row)
    return rows


def split_header(rows: List[List[str]], header_detected: bool) -> Tuple[List[str], List[List[str]]]:
    """Sépare l'en-tête du corps.

    Si PyMuPDF a détecté l'en-tête *dans* le tableau (``header.external == False``),
    la première ligne est l'en-tête. Sinon, en-têtes génériques ``col_N``.
    Les noms vides ou dupliqués sont rendus uniques (contrainte des DataFrames/Markdown).
    """
    if not rows:
        return [], []
    width = len(rows[0])
    if header_detected:
        raw_header, body = rows[0], rows[1:]
    else:
        raw_header, body = [""] * width, rows

    header: List[str] = []
    for i, name in enumerate(raw_header, 1):
        name = name or f"col_{i}"
        candidate, n = name, 2
        while candidate in header:
            candidate, n = f"{name}_{n}", n + 1
        header.append(candidate)
    return header, body


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def to_markdown(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Rendu Markdown sans dépendance (``tabulate`` découpe les cellules multilignes)."""
    lines = [
        "| " + " | ".join(_md_cell(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in body]
    return "\n".join(lines)
