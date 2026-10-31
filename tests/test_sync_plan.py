from pathlib import Path

import pytest

from notefinder import db
from notefinder.db import PageId, StoredPage
from notefinder.render import hash_image, render_page
from notefinder.sync.plan import (EmbedJob, EmbedReason, ExtractJob, ExtractReason,
                                  LocalPage, embedding_text, in_scope, make_plan,
                                  plan_sync, scan)
from notefinder.sync.target import Target
from tests.helpers import PARAMS, make_pdf

SOURCE_FILE = "lecture.pdf"
PAGE_ID = PageId(SOURCE_FILE, 1)
PDF = Path("/notes/lecture.pdf")


def local(page_num: int = 1, source_file: str = SOURCE_FILE) -> LocalPage:
    return LocalPage(PageId(source_file, page_num), "hash", PDF)


def stored(page_num: int = 1, source_file: str = SOURCE_FILE, **overrides) -> StoredPage:
    """A stored page in sync with local() and PARAMS, unless overridden."""
    fields = {
        "content_hash": "hash",
        "prompt_version": PARAMS.prompt.version,
        "claude_model": PARAMS.claude_model,
        "embedded_text": embedding_text("Summary.", ["Concept"]),
        "embed_model": PARAMS.embed_model,
        "summary": "Summary.",
        "concepts": ["Concept"],
    } | overrides
    return StoredPage(PageId(source_file, page_num), **fields)


def test_embedding_text_format():
    assert embedding_text("A summary.", ["A", "B"]) == "A summary. Concepts: A, B"


def test_a_page_in_sync_is_unchanged():
    plan = make_plan([local()], [stored()], PARAMS)

    assert (plan.extract, plan.embed, plan.delete, plan.unchanged) == ([], [], [], 1)
    assert plan.is_empty()


def test_no_pages_makes_an_empty_plan():
    assert make_plan([], [], PARAMS).is_empty()


def test_a_new_page_is_extracted():
    plan = make_plan([local()], [], PARAMS)

    assert plan.extract == [ExtractJob(PAGE_ID, PDF, ExtractReason.NEW)]
    assert not plan.is_empty()


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"content_hash": "old-hash"}, ExtractReason.PAGE_CHANGED),
    ({"prompt_version": "v1"}, ExtractReason.PROMPT_CHANGED),
    ({"claude_model": "claude-old"}, ExtractReason.CLAUDE_MODEL_CHANGED),
])
def test_a_page_is_extracted_again_when_its_extraction_is_stale(overrides, reason):
    plan = make_plan([local()], [stored(**overrides)], PARAMS)

    assert plan.extract == [ExtractJob(PAGE_ID, PDF, reason)]
    assert (plan.embed, plan.unchanged) == ([], 0)


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"embed_model": "embed-old"}, EmbedReason.EMBED_MODEL_CHANGED),
    ({"embedded_text": "Summary. | Concept"}, EmbedReason.EMBEDDING_TEXT_CHANGED),
])
def test_a_page_is_re_embedded_from_its_stored_extraction(overrides, reason):
    plan = make_plan([local()], [stored(**overrides)], PARAMS)

    assert plan.embed == [EmbedJob(PAGE_ID, "Summary.", ["Concept"], reason)]
    assert (plan.extract, plan.unchanged) == ([], 0)


def test_a_page_needing_both_is_only_extracted():
    """Extracting embeds the new results too, so a separate re-embed would be wasted."""
    page = stored(prompt_version="v1", embed_model="embed-old")

    plan = make_plan([local()], [page], PARAMS)

    assert [job.reason for job in plan.extract] == [ExtractReason.PROMPT_CHANGED]
    assert plan.embed == []


def test_a_page_change_is_reported_over_config_changes():
    page = stored(content_hash="old-hash", prompt_version="v1", claude_model="claude-old")

    plan = make_plan([local()], [page], PARAMS)

    assert [job.reason for job in plan.extract] == [ExtractReason.PAGE_CHANGED]


def test_jobs_lists_extract_jobs_then_embed_jobs():
    plan = make_plan([local(1), local(2)], [stored(1, embed_model="embed-old")], PARAMS)

    jobs = [(type(job), job.id.page_num) for job in plan.jobs]
    assert jobs == [(ExtractJob, 2), (EmbedJob, 1)]


def test_stored_pages_that_are_gone_are_deleted_in_order():
    pages = [stored(4), stored(1), stored(3), stored(1, source_file="old.pdf")]

    plan = make_plan([local(1)], pages, PARAMS)

    assert plan.delete == [
        PageId(SOURCE_FILE, 3), PageId(SOURCE_FILE, 4), PageId("old.pdf", 1),
    ]
    assert plan.unchanged == 1


def test_syncing_a_folder_deletes_only_inside_it():
    pages = [stored(1, "os/a.pdf"), stored(2, "os/a.pdf"), stored(1, "math/b.pdf")]

    plan = make_plan([local(1, "os/a.pdf")], pages, PARAMS, scope="os")

    assert plan.delete == [PageId("os/a.pdf", 2)]


def test_syncing_a_pdf_deletes_only_its_pages():
    pages = [stored(1, "os/a.pdf"), stored(2, "os/a.pdf"), stored(1, "os/b.pdf")]

    plan = make_plan([local(1, "os/a.pdf")], pages, PARAMS, scope="os/a.pdf")

    assert plan.delete == [PageId("os/a.pdf", 2)]


@pytest.mark.parametrize(("source_file", "scope", "expected"), [
    ("os/a.pdf", "", True),
    ("os/a.pdf", "os", True),
    ("os/sub/a.pdf", "os", True),
    ("os/a.pdf", "os/a.pdf", True),
    ("osx/a.pdf", "os", False),  # a folder, not a string prefix
    ("os/ab.pdf", "os/a.pdf", False),
    ("math/a.pdf", "os", False),
])
def test_in_scope(source_file, scope, expected):
    assert in_scope(source_file, scope) is expected


def test_scan_hashes_every_page_under_the_target(tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])
    pdf = make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes", "threads"])

    pages = scan(Target(tmp_path, tmp_path / "os"), PARAMS.dpi)

    assert [(p.id, p.pdf_path) for p in pages] == [
        (PageId("os/lecture1.pdf", 1), pdf), (PageId("os/lecture1.pdf", 2), pdf)]
    assert [p.content_hash for p in pages] == [
        hash_image(render_page(pdf, n, PARAMS.dpi)) for n in (1, 2)]


def test_planning_changes_nothing(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])

    plan = plan_sync(conn, Target.resolve(tmp_path), PARAMS)

    assert len(plan.extract) == 2
    assert db.get_stored_pages(conn) == []
