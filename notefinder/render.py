"""Rendering PDF pages to images, and hashing them to detect changed pages."""
import hashlib
from pathlib import Path

import pymupdf


def page_count(pdf_path: Path) -> int:
    """Count a PDF's pages.

    :param pdf_path: The PDF.
    :returns: The number of pages.
    """
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def render_page(pdf_path: Path, page_num: int, dpi: int) -> bytes:
    """Render one page to a PNG.

    :param pdf_path: The PDF.
    :param page_num: The page, 1-based. Must be from 1 to the page count.
    :param dpi: The resolution to render at.
    :returns: The PNG bytes.
    """
    with pymupdf.open(pdf_path) as doc:
        pixmap = doc[page_num - 1].get_pixmap(dpi=dpi)
        return pixmap.tobytes("png")


def hash_image(png_bytes: bytes) -> str:
    """Hash a rendered page, so a changed page can be detected.

    :param png_bytes: The PNG bytes.
    :returns: The SHA-256 hex digest.
    """
    return hashlib.sha256(png_bytes).hexdigest()
