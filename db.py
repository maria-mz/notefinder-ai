import os
from dataclasses import dataclass

import psycopg

DEFAULT_DB_URL = "postgresql://notefinder:notefinder@localhost:5432/notefinder"

PAGE_COLUMNS = """source_file, page_num, content_hash, transcript, concepts,
                  summary, prompt_version, embedding"""


@dataclass
class Page:
    source_file: str
    page_num: int
    content_hash: str
    transcript: str
    concepts: list[str]
    summary: str
    prompt_version: str
    embedding: list[float]


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or os.environ.get("NOTEFINDER_DB_URL", DEFAULT_DB_URL)
    return psycopg.connect(url, autocommit=True)


def to_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def from_pgvector(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def get_page(conn, source_file: str, page_num: int) -> Page | None:
    row = conn.execute(
        f"SELECT {PAGE_COLUMNS} FROM pages WHERE source_file = %s AND page_num = %s",
        (source_file, page_num),
    ).fetchone()
    return _row_to_page(row) if row is not None else None


def get_hashes(conn) -> dict[tuple[str, int], str]:
    """The content hash of every finished page, keyed by (source_file, page_num)."""
    rows = conn.execute("SELECT source_file, page_num, content_hash FROM pages").fetchall()
    return {(source_file, page_num): h for source_file, page_num, h in rows}


def save_page(conn, source_file: str, page_num: int, content_hash: str, *,
              transcript: str, concepts: list[str], summary: str,
              prompt_version: str, embedding: list[float]) -> None:
    """Insert a finished page, or overwrite its earlier results. One atomic statement."""
    assert (
        content_hash and transcript and concepts and summary
        and prompt_version and embedding
    ), "Results must not be empty"

    conn.execute(
        """
        INSERT INTO pages (source_file, page_num, content_hash, transcript, concepts,
                           summary, prompt_version, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (source_file, page_num) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            transcript = EXCLUDED.transcript,
            concepts = EXCLUDED.concepts,
            summary = EXCLUDED.summary,
            prompt_version = EXCLUDED.prompt_version,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        """,
        (source_file, page_num, content_hash, transcript, concepts,
         summary, prompt_version, to_pgvector(embedding)),
    )


def _row_to_page(row) -> Page:
    # the order here must match PAGE_COLUMNS
    return Page(
        source_file=row[0], page_num=row[1], content_hash=row[2],
        transcript=row[3], concepts=row[4], summary=row[5],
        prompt_version=row[6], embedding=from_pgvector(row[7]),
    )
