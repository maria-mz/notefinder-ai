"""Searching pages: semantic search, and BM25 keyword search as its baseline."""
import re

import snowballstemmer
from rank_bm25 import BM25Okapi

from notefinder import db
from notefinder.db import PageId, SearchResult

_stemmer = snowballstemmer.stemmer("english")


def search(conn, embedder, query: str, k: int) -> list[SearchResult]:
    """Find the pages nearest to a query.

    Only pages embedded with ``embedder``'s model are searched.

    :param conn: The database connection.
    :param embedder: Embeds the query, and names the model to match stored pages on.
    :param query: The search text, e.g. a topic name.
    :param k: The most results to return.
    :returns: Up to ``k`` pages, closest first by cosine similarity.
    """
    return db.search(
        conn, embedder.embed_query(query), limit=k, embed_model=embedder.model_name
    )


def tokenize(text: str) -> list[str]:
    """Split text into stemmed, lowercase runs of letters and digits.

    :param text: The text to split.
    :returns: The tokens, in order, e.g. ``["hass", "diagram"]`` for ``"Hasse diagrams"``.
    """
    return _stemmer.stemWords(re.findall(r"[^\W_]+", text.lower()))


class Bm25Index:
    """A BM25 index over pages' text.

    :ivar page_ids: The indexed pages, sorted.
    :ivar bm25: The index, with one document per page in ``page_ids`` order.
    """
    def __init__(self, texts: dict[PageId, str]):
        """Index each page's text.

        :param texts: The text to index for each page. Must not be empty.
        """
        self.page_ids = sorted(texts)
        self.bm25 = BM25Okapi([tokenize(texts[p]) for p in self.page_ids])

    def rank(self, query: str) -> list[PageId]:
        """Rank pages by their BM25 score for a query.

        :param query: The search text.
        :returns: The pages scoring above 0, best first. Ties go to page order.
        """
        scores = self.bm25.get_scores(tokenize(query))
        scored = [
            (score, page_id)
            for score, page_id in zip(scores, self.page_ids)
            if score > 0
        ]
        sort_key = lambda s: (-s[0], s[1])
        return [page_id for score, page_id in sorted(scored, key=sort_key)]
