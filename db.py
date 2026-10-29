import os
from enum import Enum, auto
from dataclasses import dataclass

import psycopg

DEFAULT_DB_URL = "postgresql://notefinder:notefinder@localhost:5432/notefinder"


@dataclass
class Page:
    source_file: str
    page_num: int
    content_hash: str

    status: str = "pending"
    attempts: int = 0
    last_error: str | None = None

    transcript: str | None = None
    concepts: list[str] | None = None
    summary: str | None = None
    prompt_version: str | None = None
    embedding: list[float] | None = None


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or os.environ.get("NOTEFINDER_DB_URL", DEFAULT_DB_URL)
    return psycopg.connect(url, autocommit=True)


def to_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def from_pgvector(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


class UpsertResult(Enum):
    # The page didn't exist and was inserted
    NEW = auto()

    # The page's content hash changed, so its status was reset to pending
    # its results were cleared
    CHANGED = auto()

    # The page already existed with the same content hash
    UNCHANGED = auto()


def upsert_page(conn, source_file: str, page_num: int,
                content_hash: str) -> UpsertResult:
    """Insert a page or synchronize it with its current image."""

    with conn.transaction():
        row = conn.execute(
            """
            SELECT content_hash FROM pages
            WHERE source_file = %s AND page_num = %s
            FOR UPDATE
            """,
            (source_file, page_num),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO pages (source_file, page_num, content_hash)
                VALUES (%s, %s, %s)
                """,
                (source_file, page_num, content_hash),
            )
            return UpsertResult.NEW

        if row[0] == content_hash:
            return UpsertResult.UNCHANGED

        conn.execute(
            """
            UPDATE pages
            SET content_hash = %s, status = 'pending', attempts = 0,
                last_error = NULL, transcript = NULL, concepts = NULL,
                summary = NULL, prompt_version = NULL, embedding = NULL,
                updated_at = now()
            WHERE source_file = %s AND page_num = %s
            """,
            (content_hash, source_file, page_num),
        )
        return UpsertResult.CHANGED


def get_page(conn, source_file: str, page_num: int) -> Page | None:
    """Return the page if it exists, otherwise None"""

    row = conn.execute(
        """
        SELECT source_file, page_num, content_hash,
               status, attempts, last_error,
               transcript, concepts, summary, prompt_version, embedding
        FROM pages
        WHERE source_file = %s AND page_num = %s
        """,
        (source_file, page_num),
    ).fetchone()

    return _row_to_page(row) if row is not None else None


def claim_page(conn, source_file: str, page_num: int) -> Page | None:
    """
    Claim a pending page by transitioning it to processing.

    Returns the updated Page, or None if the page wasn't pending.
    """
    row = conn.execute(
        """
        UPDATE pages
        SET status = 'processing',
            attempts = attempts + 1,
            last_error = NULL,
            updated_at = now()
        WHERE source_file = %s AND page_num = %s AND status = 'pending'
        RETURNING source_file, page_num, content_hash,
                  status, attempts, last_error,
                  transcript, concepts, summary, prompt_version, embedding
        """,
        (source_file, page_num),
    ).fetchone()

    return _row_to_page(row) if row is not None else None


def mark_done(conn, source_file: str, page_num: int, *, transcript: str,
              concepts: list[str], summary: str, prompt_version: str,
              embedding: list[float]) -> Page:
    """
    Mark a processing page as done.

    Returns the updated Page.
    """
    assert (
        transcript
        and concepts
        and summary
        and prompt_version
        and embedding
    ), "Results must not be empty"

    row = conn.execute(
        """
        UPDATE pages
        SET status = 'done', transcript = %s, concepts = %s, summary = %s,
            prompt_version = %s, embedding = %s::vector, last_error = NULL,
            updated_at = now()
        WHERE source_file = %s AND page_num = %s AND status = 'processing'
        RETURNING source_file, page_num, content_hash,
                  status, attempts, last_error,
                  transcript, concepts, summary, prompt_version, embedding
        """,
        (
            transcript,
            concepts,
            summary,
            prompt_version,
            to_pgvector(embedding),
            source_file,
            page_num,
        ),
    ).fetchone()

    assert row, f"{source_file} p.{page_num} not in 'processing'"

    return _row_to_page(row)


def mark_failed(conn, source_file: str, page_num: int, error: str) -> Page:
    """
    Mark a processing page as failed.

    Returns the updated Page.
    """
    row = conn.execute(
        """
        UPDATE pages
        SET status = 'failed', last_error = %s, updated_at = now()
        WHERE source_file = %s AND page_num = %s and status = 'processing'
        RETURNING source_file, page_num, content_hash,
                  status, attempts, last_error,
                  transcript, concepts, summary, prompt_version, embedding
        """,
        (error, source_file, page_num),
    ).fetchone()

    assert row, f"{source_file} p.{page_num} not in 'processing'"

    return _row_to_page(row)


def reset_stuck(conn) -> int:
    """Reset all processing pages to pending after a crashed run"""
    cur = conn.execute(
        """
        UPDATE pages
        SET status = 'pending', updated_at = now()
        WHERE status = 'processing'
        """
    )
    return cur.rowcount


def _row_to_page(row) -> Page:
    return Page(
        source_file=row[0],
        page_num=row[1],
        content_hash=row[2],
        status=row[3],
        attempts=row[4],
        last_error=row[5],
        transcript=row[6],
        concepts=row[7],
        summary=row[8],
        prompt_version=row[9],
        embedding=from_pgvector(row[10]) if row[10] is not None else None,
    )
