"""Storing pages in Postgres with pgvector: extractions, embeddings, and vector search."""
from dataclasses import dataclass

import psycopg

PAGE_COLUMNS = """source_file, page_num, content_hash, transcript, concepts, summary,
                  prompt_version, claude_model, embedded_text, embed_model, embedding"""


@dataclass(frozen=True, order=True)
class PageId:
    """Which page of which file: the key of a stored page.

    :ivar source_file: The PDF's path, relative to the notes folder.
    :ivar page_num: The page, 1-based.
    """
    source_file: str
    page_num: int

    def __str__(self) -> str:
        return f"{self.source_file} p.{self.page_num}"


@dataclass
class Page:
    """A stored page, with everything derived from it.

    :ivar id: Which page this is.
    :ivar content_hash: SHA-256 of the page image Claude read.
    :ivar transcript: The page's text, transcribed.
    :ivar concepts: The topics the page covers.
    :ivar summary: A short summary of the page.
    :ivar prompt_version: The version of the prompt the page was extracted with.
    :ivar claude_model: The Claude model that extracted the page.
    :ivar embedded_text: The exact text that was embedded.
    :ivar embed_model: The model that embedded ``embedded_text``.
    :ivar embedding: The vector for ``embedded_text``.
    """
    id: PageId
    content_hash: str
    transcript: str
    concepts: list[str]
    summary: str
    prompt_version: str
    claude_model: str
    embedded_text: str
    embed_model: str
    embedding: list[float]


@dataclass
class StoredPage:
    """A stored page without its transcript and vector.

    Those are large, and not needed to plan a sync or check an eval. The fields are
    as in :class:`Page`.
    """
    id: PageId
    content_hash: str
    prompt_version: str
    claude_model: str
    embedded_text: str
    embed_model: str
    summary: str
    concepts: list[str]


@dataclass
class SearchResult:
    """A page found by :func:`search`.

    :ivar id: Which page this is.
    :ivar summary: A short summary of the page.
    :ivar concepts: The topics the page covers.
    :ivar score: Cosine similarity to the query, higher is closer.
    """
    id: PageId
    summary: str
    concepts: list[str]
    score: float


def connect(url: str) -> psycopg.Connection:
    """Open a database connection in autocommit mode.

    :param url: The Postgres connection URL.
    :returns: The connection.
    """
    return psycopg.connect(url, autocommit=True)


def to_pgvector(vector: list[float]) -> str:
    """Format a vector as pgvector text, e.g. ``"[0.1,0.2]"``.

    :param vector: The vector.
    :returns: The text, to cast to ``vector`` in SQL.
    """
    return "[" + ",".join(str(x) for x in vector) + "]"


def from_pgvector(text: str) -> list[float]:
    """Parse pgvector text, e.g. ``"[0.1,0.2]"``, into a vector.

    :param text: The text, as Postgres returns a ``vector``.
    :returns: The vector.
    """
    return [float(x) for x in text.strip("[]").split(",")]


def get_page(conn, page_id: PageId) -> Page | None:
    """Fetch one stored page.

    :param conn: The database connection.
    :param page_id: The page to fetch.
    :returns: The page, or ``None`` if it isn't found.
    """
    row = conn.execute(
        f"SELECT {PAGE_COLUMNS} FROM pages WHERE source_file = %s AND page_num = %s",
        (page_id.source_file, page_id.page_num),
    ).fetchone()
    return _row_to_page(row) if row is not None else None


def get_stored_pages(conn) -> list[StoredPage]:
    """Fetch every stored page, without transcripts and vectors.

    :param conn: The database connection.
    :returns: The pages, in no particular order.
    """
    rows = conn.execute(
        """
        SELECT source_file, page_num, content_hash, prompt_version, claude_model,
               embedded_text, embed_model, summary, concepts
        FROM pages
        """
    ).fetchall()
    return [StoredPage(PageId(row[0], row[1]), *row[2:]) for row in rows]


def get_transcripts(conn) -> dict[PageId, str]:
    """Fetch every stored page's transcript.

    :param conn: The database connection.
    :returns: Each page's transcript, by page.
    """
    rows = conn.execute("SELECT source_file, page_num, transcript FROM pages").fetchall()
    return {PageId(row[0], row[1]): row[2] for row in rows}


def save_page(conn, page_id: PageId, content_hash: str, *,
              transcript: str, concepts: list[str], summary: str,
              prompt_version: str, claude_model: str,
              embedded_text: str, embed_model: str, embedding: list[float]) -> None:
    """Insert a finished page, or overwrite its earlier results as one atomic statement.

    :param conn: The database connection.
    :param page_id: The page to save.
    :param content_hash: SHA-256 of the page image Claude read.
    :raises psycopg.Error: If any argument after ``page_id`` is empty, which the
        schema rejects.
    """
    conn.execute(
        """
        INSERT INTO pages (source_file, page_num, content_hash,
                           transcript, concepts, summary,
                           prompt_version, claude_model,
                           embedded_text, embed_model, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (source_file, page_num) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            transcript = EXCLUDED.transcript,
            concepts = EXCLUDED.concepts,
            summary = EXCLUDED.summary,
            prompt_version = EXCLUDED.prompt_version,
            claude_model = EXCLUDED.claude_model,
            embedded_text = EXCLUDED.embedded_text,
            embed_model = EXCLUDED.embed_model,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        """,
        (page_id.source_file, page_id.page_num, content_hash,
         transcript, concepts, summary,
         prompt_version, claude_model,
         embedded_text, embed_model, to_pgvector(embedding)),
    )


def save_embedding(conn, page_id: PageId, *,
                   embedded_text: str, embed_model: str, embedding: list[float]) -> None:
    """Replace a stored page's embedding, keeping its extraction.

    Does nothing if the page isn't stored.

    :param conn: The database connection.
    :param page_id: The page to update.
    :param embedded_text: The exact text that was embedded.
    :param embed_model: The model that embedded ``embedded_text``.
    :param embedding: The vector for ``embedded_text``.
    :raises psycopg.Error: If any argument after ``page_id`` is empty, which the
        schema rejects.
    """
    conn.execute(
        """
        UPDATE pages SET
            embedded_text = %s,
            embed_model = %s,
            embedding = %s::vector,
            updated_at = now()
        WHERE source_file = %s AND page_num = %s
        """,
        (embedded_text, embed_model, to_pgvector(embedding),
         page_id.source_file, page_id.page_num),
    )


def delete_pages(conn, page_ids: list[PageId]) -> None:
    """Delete the pages with these ids, as an atomic operation.

    Ids that aren't stored are skipped.

    :param conn: The database connection.
    :param page_ids: The pages to delete.
    """
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany("DELETE FROM pages WHERE source_file = %s AND page_num = %s",
                        [(p.source_file, p.page_num) for p in page_ids])


def search(conn, query_vector: list[float], limit: int,
           embed_model: str) -> list[SearchResult]:
    """The pages nearest to query_vector by cosine distance, best first.

    Only pages embedded with embed_model are searched: vectors from different
    models can't be compared. An exact scan: no vector index, so results are
    never approximate. Ties are broken by page id, so the order is stable.

    :param conn: The database connection.
    :param query_vector: The embedded query.
    :param limit: The most results to return.
    :param embed_model: The model that embedded ``query_vector``.
    :returns: Up to ``limit`` pages, closest first.
    """
    vector = to_pgvector(query_vector)
    rows = conn.execute(
        """
        SELECT source_file, page_num, summary, concepts,
               1 - (embedding <=> %s::vector) AS score
        FROM pages
        WHERE embed_model = %s
        ORDER BY embedding <=> %s::vector, source_file, page_num
        LIMIT %s
        """,
        (vector, embed_model, vector, limit),
    ).fetchall()
    return [SearchResult(PageId(row[0], row[1]), *row[2:]) for row in rows]


def _row_to_page(row) -> Page:
    """Build a :class:`Page` from a row selected with :data:`PAGE_COLUMNS`."""
    # the order here must match PAGE_COLUMNS
    return Page(
        id=PageId(row[0], row[1]), content_hash=row[2],
        transcript=row[3], concepts=row[4], summary=row[5],
        prompt_version=row[6], claude_model=row[7],
        embedded_text=row[8], embed_model=row[9], embedding=from_pgvector(row[10]),
    )
