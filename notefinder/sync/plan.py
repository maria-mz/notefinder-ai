"""Deciding what a sync has to do.

:func:`make_plan` is pure: no database, files, or API calls. :func:`plan_sync`
gathers its inputs.
"""
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from notefinder import db
from notefinder.db import PageId, PageState
from notefinder.prompt import Prompt
from notefinder.render import hash_image, page_count, render_page
from notefinder.sync.target import Target


class ExtractReason(StrEnum):
    NEW = "new"
    PAGE_CHANGED = "page changed"
    PROMPT_CHANGED = "prompt changed"
    CLAUDE_MODEL_CHANGED = "Claude model changed"


class EmbedReason(StrEnum):
    EMBED_MODEL_CHANGED = "embedding model changed"
    EMBEDDING_TEXT_CHANGED = "embedding text changed"


@dataclass(frozen=True)
class SyncParams:
    """The settings a sync runs with.

    Which of them make stored pages stale is decided by :func:`extract_reason`
    and :func:`embed_reason`, not by this class.
    """
    prompt: Prompt
    claude_model: str
    embed_model: str
    dpi: int
    num_workers: int


@dataclass
class LocalPage:
    id: PageId
    content_hash: str
    pdf_path: Path


@dataclass
class ExtractJob:
    """Read the page with Claude, then embed it."""
    id: PageId
    pdf_path: Path
    reason: ExtractReason


@dataclass
class EmbedJob:
    """Embed the page again from its stored extraction. No API call."""
    id: PageId
    summary: str
    concepts: list[str]
    reason: EmbedReason


Job = ExtractJob | EmbedJob


@dataclass
class Plan:
    extract: list[ExtractJob]
    embed: list[EmbedJob]
    delete: list[PageId]
    unchanged: int

    @property
    def jobs(self) -> list[Job]:
        return [*self.extract, *self.embed]

    def is_empty(self) -> bool:
        return not (self.extract or self.embed or self.delete)


def embedding_text(summary: str, concepts: list[str]) -> str:
    """What gets embedded. Pages whose stored text differs from this get re-embedded."""
    return f"{summary} Concepts: {', '.join(concepts)}"


def in_scope(source_file: str, scope: str) -> bool:
    """Whether source_file is the PDF, or under the folder, named by scope ('' is all of them)."""
    return scope == "" or source_file == scope or source_file.startswith(scope + "/")


# When a stored page is stale:
# - Re-extract if it's new, its rendered image changed, or it was extracted with
#   another prompt or Claude model. The image hash catches anything that shapes the
#   image: the PDF itself, dpi, and the PyMuPDF version.
# - Otherwise re-embed if it was embedded with another model, or its embedding text
#   format changed (compared against the stored text).
# - Not tracked: sentence-transformers and torch versions, and max_tokens.
#
# A page can be stale for several reasons at once, but each job reports one: the
# first that applies. The order is a priority, content before settings, so after a
# settings change the plan still shows which pages' content also changed.
def extract_reason(page: LocalPage, stored: PageState | None,
                   params: SyncParams) -> ExtractReason | None:
    if stored is None:
        return ExtractReason.NEW
    if stored.content_hash != page.content_hash:
        return ExtractReason.PAGE_CHANGED
    if stored.prompt_version != params.prompt.version:
        return ExtractReason.PROMPT_CHANGED
    if stored.claude_model != params.claude_model:
        return ExtractReason.CLAUDE_MODEL_CHANGED
    return None


def embed_reason(stored: PageState, params: SyncParams) -> EmbedReason | None:
    if stored.embed_model != params.embed_model:
        return EmbedReason.EMBED_MODEL_CHANGED
    if stored.embedded_text != embedding_text(stored.summary, stored.concepts):
        return EmbedReason.EMBEDDING_TEXT_CHANGED
    return None


def make_plan(
    local_pages: list[LocalPage],
    stored_pages: list[PageState],
    params: SyncParams,
    scope: str = ""
) -> Plan:
    """What it takes for the stored pages under scope to match local_pages and params.

    local_pages must be every page under scope: stored pages under scope that are
    missing from it get deleted.
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
    """Every page the target covers, with the hash of its rendered image."""
    pages: list[LocalPage] = []

    for pdf in target.pdfs():
        source_file = target.source_file(pdf)

        for page in range(1, page_count(pdf) + 1):
            content_hash = hash_image(render_page(pdf, page, dpi))
            pages.append(LocalPage(PageId(source_file, page), content_hash, pdf))

    return pages


def plan_sync(conn, target: Target, params: SyncParams) -> Plan:
    """What syncing target would do. Changes nothing."""
    pages = scan(target, params.dpi)
    return make_plan(pages, db.get_page_states(conn), params, target.scope)
