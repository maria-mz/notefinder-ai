from pathlib import Path
from typing import Callable

import pymupdf

from notefinder import db
from notefinder.db import PageId
from notefinder.extract import Extraction
from notefinder.prompt import Prompt
from notefinder.sync.plan import SyncParams, embedding_text

PARAMS = SyncParams(
    prompt=Prompt("Transcribe this page."),
    claude_model="mock-claude",
    embed_model="mock-embed",
    dpi=72,
    num_workers=2,
)


def constant_extraction(png_bytes: bytes) -> Extraction:
    return Extraction(
        transcript="Transcript",
        concepts=["Concept", "Another concept"],
        summary="Summary",
        input_tokens=100,
        output_tokens=50,
    )


def mock_vector(text: str) -> list[float]:
    """A deterministic 768-number 'embedding'."""
    return [(sum(map(ord, text)) % 997) / 1000] * 768


def axis_vector(*weights: float) -> list[float]:
    """A 768-number vector with the given leading values and zeros after."""
    return [*weights, *([0.0] * (768 - len(weights)))]


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    """A PDF with one page per text, so pages render (and hash) differently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        for text in page_texts:
            doc.new_page().insert_text((72, 72), text)
        doc.save(path)
    return path


def save_test_page(conn, page_id: PageId = PageId("lecture.pdf", 1), **overrides) -> None:
    """Save a page in sync with PARAMS. Any save_page argument can be overridden."""
    fields = {
        "content_hash": "hash",
        "transcript": "Transcript",
        "concepts": ["Concept"],
        "summary": "Summary",
        "prompt_version": PARAMS.prompt.version,
        "claude_model": PARAMS.claude_model,
        "embedded_text": embedding_text("Summary", ["Concept"]),
        "embed_model": PARAMS.embed_model,
        "embedding": axis_vector(1, 0),
    } | overrides
    db.save_page(conn, page_id, fields.pop("content_hash"), **fields)


class MockExtractor:
    def __init__(
        self,
        extract: Callable[[bytes], Extraction] = constant_extraction,
        fail: bool = False,
    ):
        self.extract_fn = extract
        self.fail = fail

    async def extract(self, png_bytes: bytes, prompt: Prompt) -> Extraction:
        if self.fail:
            raise RuntimeError("API down")
        return self.extract_fn(png_bytes)


class MockEmbedder:
    def __init__(
        self,
        embed: Callable[[str], list[float]] = mock_vector,
        fail: bool = False,
        model_name: str = PARAMS.embed_model,
    ):
        self.embed_fn = embed
        self.fail = fail
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed")
        return [self.embed_fn(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        if self.fail:
            raise RuntimeError("embedding failed")
        return self.embed_fn(query)
