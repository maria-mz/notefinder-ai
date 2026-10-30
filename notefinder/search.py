from notefinder import db
from notefinder.db import SearchResult


def search(conn, embedder, query: str, k: int) -> list[SearchResult]:
    """The k pages nearest to the query. The CLI and the eval both search through here."""
    return db.search(conn, embedder.embed_query(query), limit=k, embed_model=embedder.model_name)
