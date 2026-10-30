class Embedder:
    def __init__(self, model_name: str, query_prefix: str):
        # sentence-transformers takes seconds to import; importing it here keeps
        # importing this module cheap
        from huggingface_hub.utils import logging as hf_logging
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        hf_logging.set_verbosity_error()  # warns twice per run when HF_TOKEN isn't set
        transformers_logging.disable_progress_bar()  # draws a bar while loading weights

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
