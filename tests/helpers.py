from typing import Callable

from extract import Extraction


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


class MockExtractor:
    def __init__(
        self,
        extract: Callable[[bytes], Extraction] = constant_extraction,
        fail: bool = False,
    ):
        self.extract_fn = extract
        self.fail = fail

    async def extract(self, png_bytes: bytes, prompt: str) -> Extraction:
        if self.fail:
            raise RuntimeError("API down")
        return self.extract_fn(png_bytes)


class MockEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed")
        return [mock_vector(text) for text in texts]
