from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import fitz  # PyMuPDF


@lru_cache(maxsize=64)
def _open_pdf(pdf_path: str) -> fitz.Document:
    # caching opened PDF objects speeds repeated page renders
    return fitz.open(pdf_path)


def render_pdf_page_to_png_bytes(
    pdf_path: str,
    page_num_1based: int,
    zoom: float = 2.0,
) -> bytes:
    """
    Render a given PDF page to PNG bytes.
    page_num_1based: 1..N (human-friendly)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = _open_pdf(pdf_path)
    page_index = page_num_1based - 1
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(f"Page out of range: {page_num_1based} (PDF has {len(doc)} pages)")

    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False => smaller files
    return pix.tobytes("png")


def get_pdf_page_count(pdf_path: str) -> int:
    doc = _open_pdf(pdf_path)
    return len(doc)
