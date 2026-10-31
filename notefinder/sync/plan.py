"""Deciding what a sync has to do."""
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from notefinder import db
from notefinder.db import PageId, StoredPage
from notefinder.prompt import Prompt
from notefinder.render import hash_image, page_count, render_page
from notefinder.sync.target import Target


class ExtractReason(StrEnum):
    """Why a page must be read with Claude again."""
    NEW = "new"
    PAGE_CHANGED = "page changed"
    PROMPT_CHANGED = "prompt changed"
    CLAUDE_MODEL_CHANGED = "Claude model changed"


class EmbedReason(StrEnum):
    """Why a page must be embedded again."""
    EMBED_MODEL_CHANGED = "embedding model changed"
    EMBEDDING_TEXT_CHANGED = "embedding text changed"


@dataclass(frozen=True)
class SyncParams:
    """The settings a sync runs with.

    :ivar prompt: The extraction prompt.
    :ivar claude_model: The Claude model that reads pages.
    :ivar embed_model: The embedding model.
    :ivar dpi: The resolution pages are rendered at.
    :ivar num_workers: How many pages a sync processes at once.
    """
    prompt: Prompt
    claude_model: str
    embed_model: str
    dpi: int
    num_workers: int


@dataclass
class LocalPage:
    """A page as it is now in the notes folder.

    :ivar id: Which page this is.
    :ivar content_hash: SHA-256 of the rendered page image.
    :ivar pdf_path: The PDF the page is in. Absolute.
    """
    id: PageId
    content_hash: str
    pdf_path: Path


@dataclass
class ExtractJob:
    """Read the page with Claude, then embed it.

    :ivar id: The page.
    :ivar pdf_path: The PDF the page is in. Absolute.
    :ivar reason: Why the page needs it.
    """
    id: PageId
    pdf_path: Path
    reason: ExtractReason


@dataclass
class EmbedJob:
    """Embed the page again from its stored extraction.

    :ivar id: The page.
    :ivar summary: The stored summary, to embed.
    :ivar concepts: The stored concepts, to embed.
    :ivar reason: Why the page needs it.
    """
    id: PageId
    summary: str
    concepts: list[str]
    reason: EmbedReason


Job = ExtractJob | EmbedJob


@dataclass
class Plan:
    """What a sync will do.

    :ivar extract: Pages to read with Claude and embed.
    :ivar embed: Pages to embed again from their stored extraction.
    :ivar delete: Stored pages no longer in the notes.
    :ivar unchanged: How many pages are already up to date.
    """
    extract: list[ExtractJob]
    embed: list[EmbedJob]
    delete: list[PageId]
    unchanged: int

    @property
    def jobs(self) -> list[Job]:
        """Every extract job, then every embed job."""
        return [*self.extract, *self.embed]

    def is_empty(self) -> bool:
        """Whether the sync has nothing to change.

        :returns: ``True`` if there is nothing to extract, embed or delete.
        """
        return not (self.extract or self.embed or self.delete)


def embedding_text(summary: str, concepts: list[str]) -> str:
    """What gets embedded.

    :param summary: The page's summary.
    :param concepts: The page's concepts.
    :returns: The text to embed.
    """
    return f"{summary} Concepts: {', '.join(concepts)}"


def in_scope(source_file: str, scope: str) -> bool:
    """Whether ``source_file`` is the PDF, or under the folder, named by ``scope``."""
    return scope == "" or source_file == scope or source_file.startswith(scope + "/")


def extract_reason(page: LocalPage, stored: StoredPage | None,
                   params: SyncParams) -> ExtractReason | None:
    """Why a page must be read with Claude again, if it must.

    :param page: The page as it is now.
    :param stored: The page as stored, or ``None`` if it isn't.
    :param params: The settings to sync with.
    :returns: The first reason that applies, or ``None`` if no extraction is
        needed.
    """
    if stored is None:
        return ExtractReason.NEW
    if stored.content_hash != page.content_hash:
        return ExtractReason.PAGE_CHANGED
    if stored.prompt_version != params.prompt.version:
        return ExtractReason.PROMPT_CHANGED
    if stored.claude_model != params.claude_model:
        return ExtractReason.CLAUDE_MODEL_CHANGED
    return None


def embed_reason(stored: StoredPage, params: SyncParams) -> EmbedReason | None:
    """Why a stored page must be embedded again, if it must.

    :param stored: The page as stored.
    :param params: The settings to sync with.
    :returns: The first reason that applies, or ``None`` if no embedding is
        needed.
    """
    if stored.embed_model != params.embed_model:
        return EmbedReason.EMBED_MODEL_CHANGED
    if stored.embedded_text != embedding_text(stored.summary, stored.concepts):
        return EmbedReason.EMBEDDING_TEXT_CHANGED
    return None


def make_plan(
    local_pages: list[LocalPage],
    stored_pages: list[StoredPage],
    params: SyncParams,
    scope: str = ""
) -> Plan:
    """What it takes for the stored pages under scope to match local_pages and params.

    :param local_pages: Every page under ``scope``.
    :param stored_pages: Every stored page. Those outside ``scope`` are left alone.
    :param params: The settings to sync with.
    :param scope: The PDF or folder to sync, relative to the notes folder.
    :returns: The plan.
    """
    stored = {s.id: s for s in stored_pages}
    plan = Plan(extract=[], embed=[], delete=[], unchanged=0)

    for page in local_pages:
        row = stored.get(page.id)
        if reason := extract_reason(page, row, params):
            job = ExtractJob(page.id, page.pdf_path, reason)
            plan.extract.append(job)
        elif reason := embed_reason(row, params):
            job = EmbedJob(row.id, row.summary, row.concepts, reason)
            plan.embed.append(job)
        else:
            plan.unchanged += 1

    local_ids = {p.id for p in local_pages}
    plan.delete = sorted(
        page_id for page_id in stored
        if in_scope(page_id.source_file, scope) and page_id not in local_ids
    )
    return plan


def scan(target: Target, dpi: int) -> list[LocalPage]:
    """Every page the target covers, with the hash of its rendered image.

    Renders every page to hash it, so the cost grows with the page count.

    :param target: What to scan.
    :param dpi: The resolution to render at.
    :returns: The pages, by file then page number.
    """
    pages: list[LocalPage] = []

    for pdf in target.pdfs():
        source_file = target.source_file(pdf)

        for page in range(1, page_count(pdf) + 1):
            content_hash = hash_image(render_page(pdf, page, dpi))
            pages.append(LocalPage(PageId(source_file, page), content_hash, pdf))

    return pages


def plan_sync(conn, target: Target, params: SyncParams) -> Plan:
    """What syncing target would do. Changes nothing.

    :param conn: The database connection.
    :param target: The PDF or folder to sync.
    :param params: The settings to sync with.
    :returns: The plan.
    """
    pages = scan(target, params.dpi)
    return make_plan(pages, db.get_stored_pages(conn), params, target.scope)
