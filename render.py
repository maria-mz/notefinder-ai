import hashlib
from pathlib import Path

import pymupdf

DPI = 150


def page_count(pdf_path: Path) -> int:
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def render_page(pdf_path: Path, page_num: int, dpi: int = DPI) -> bytes:
    """Render one page (1-based) to PNG bytes."""
    with pymupdf.open(pdf_path) as doc:
        pixmap = doc[page_num - 1].get_pixmap(dpi=dpi)
        return pixmap.tobytes("png")


def hash_image(png_bytes: bytes) -> str:
    return hashlib.sha256(png_bytes).hexdigest()


def relative_source(pdf_path: Path, notes_root: Path) -> str:
    """The value stored in pages.source_file, e.g. 'os/lecture3.pdf'."""
    return pdf_path.relative_to(notes_root).as_posix()