from dataclasses import dataclass

import psycopg

PAGE_COLUMNS = """source_file, page_num, content_hash, transcript, concepts, summary,
                  prompt_version, claude_model, embedded_text, embed_model, embedding"""


@dataclass(frozen=True, order=True)
class PageId:
    """Which page of which file: the key of a stored page."""
    source_file: str
    page_num: int

    def __str__(self) -> str:
        return f"{self.source_file} p.{self.page_num}"


@dataclass
class Page:
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
class PageState:
    """What a sync needs to know about a stored page: everything but the transcript and vector."""
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
    id: PageId
    summary: str
    concepts: list[str]
    score: float  # cosine similarity, higher is closer


def connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url, autocommit=True)


def to_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def from_pgvector(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def get_page(conn, page_id: PageId) -> Page | None:
    row = conn.execute(
        f"SELECT {PAGE_COLUMNS} FROM pages WHERE source_file = %s AND page_num = %s",
        (page_id.source_file, page_id.page_num),
    ).fetchone()
    return _row_to_page(row) if row is not None else None


def get_page_states(conn) -> list[PageState]:
    rows = conn.execute(
        """
        SELECT source_file, page_num, content_hash, prompt_version, claude_model,
               embedded_text, embed_model, summary, concepts
        FROM pages
        """
    ).fetchall()
    return [PageState(PageId(row[0], row[1]), *row[2:]) for row in rows]


def save_page(conn, page_id: PageId, content_hash: str, *,
              transcript: str, concepts: list[str], summary: str,
              prompt_version: str, claude_model: str,
              embedded_text: str, embed_model: str, embedding: list[float]) -> None:
    """Insert a finished page, or overwrite its earlier results. One atomic statement."""
    assert (
        content_hash and transcript and concepts and summary
        and prompt_version and claude_model
        and embedded_text and embed_model and embedding
    ), "Results must not be empty"

    conn.execute(
        """
        INSERT INTO pages (source_file, page_num, content_hash, transcript, concepts, summary,
                           prompt_version, claude_model, embedded_text, embed_model, embedding)
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
        (page_id.source_file, page_id.page_num, content_hash, transcript, concepts, summary,
         prompt_version, claude_model, embedded_text, embed_model, to_pgvector(embedding)),
    )


def save_embedding(conn, page_id: PageId, *,
                   embedded_text: str, embed_model: str, embedding: list[float]) -> None:
    """Replace a stored page's embedding, keeping its extraction."""
    assert embedded_text and embed_model and embedding, "Results must not be empty"

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
    """Delete the pages with these ids, all or none."""
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany("DELETE FROM pages WHERE source_file = %s AND page_num = %s",
                        [(p.source_file, p.page_num) for p in page_ids])


def search(conn, query_vector: list[float], limit: int, embed_model: str) -> list[SearchResult]:
    """The pages nearest to query_vector by cosine distance, best first.

    Only pages embedded with embed_model are searched: vectors from different
    models can't be compared. An exact scan: no vector index, so results are
    never approximate.
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
    # the order here must match PAGE_COLUMNS
    return Page(
        id=PageId(row[0], row[1]), content_hash=row[2],
        transcript=row[3], concepts=row[4], summary=row[5],
        prompt_version=row[6], claude_model=row[7],
        embedded_text=row[8], embed_model=row[9], embedding=from_pgvector(row[10]),
    )
