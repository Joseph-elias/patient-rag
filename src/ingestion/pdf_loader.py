from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


def read_pdf_pages(pdf_path: str) -> List[PageText]:
    """
    Extract text page-by-page from a PDF.
    Returns a list of PageText(page_number, text).
    """
    reader = PdfReader(pdf_path)
    pages: List[PageText] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # normalize whitespace a bit
        text = " ".join(text.split())
        pages.append(PageText(page_number=i + 1, text=text))

    return pages
