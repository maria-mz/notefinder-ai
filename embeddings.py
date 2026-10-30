from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, use_query_prefix: bool = True):
        self.model = SentenceTransformer(model_name)
        self.use_query_prefix = use_query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed pages for storage."""
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query. Appended to the query prefix if configured."""
        if self.use_query_prefix:
            query = QUERY_PREFIX + query
        return self._encode([query])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
