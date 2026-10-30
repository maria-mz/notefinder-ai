class Embedder:
    def __init__(self, model_name: str, query_prefix: str):
        # sentence-transformers takes seconds to import; importing it here keeps
        # importing this module cheap
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed pages for storage."""
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query, after the query prefix ('' for none)."""
        return self._encode([self.query_prefix + query])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
