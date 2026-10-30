from dataclasses import replace

import pytest

from notefinder import db
from notefinder.db import PageId
from notefinder.sync.apply import apply_plan
from notefinder.sync.plan import EmbedReason, ExtractReason, plan_sync, scan
from notefinder.render import hash_image, render_page
from notefinder.prompt import Prompt
from notefinder.sync.target import Target
from tests.helpers import PARAMS, MockEmbedder, MockExtractor, make_pdf


def run_sync(conn, root, params=PARAMS, extractor=None, target=None):
    plan = plan_sync(conn, Target.resolve(root, target), params)
    return apply_plan(conn, plan, params, extractor or MockExtractor(),
                      MockEmbedder(model_name=params.embed_model))


def saved_pages(conn) -> list[tuple[str, int]]:
    return conn.execute(
        "SELECT source_file, page_num FROM pages ORDER BY source_file, page_num"
    ).fetchall()


def counts(summary) -> tuple[int, int, int, int, int]:
    return (summary.extracted, summary.embedded, summary.deleted, summary.unchanged,
            len(summary.failed))


def test_scan_hashes_every_page_under_the_target(tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])
    pdf = make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes", "threads"])

    pages = scan(Target(tmp_path, tmp_path / "os"), PARAMS.dpi)

    assert [(p.id, p.pdf_path) for p in pages] == [
        (PageId("os/lecture1.pdf", 1), pdf), (PageId("os/lecture1.pdf", 2), pdf)]
    assert [p.content_hash for p in pages] == [
        hash_image(render_page(pdf, 1, PARAMS.dpi)), hash_image(render_page(pdf, 2, PARAMS.dpi))]


def test_planning_changes_nothing(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])

    plan = plan_sync(conn, Target.resolve(tmp_path), PARAMS)

    assert len(plan.extract) == 2
    assert saved_pages(conn) == []


def test_sync_saves_every_page(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes"])

    summary = run_sync(conn, tmp_path)

    assert counts(summary) == (3, 0, 0, 0, 0)
    assert saved_pages(conn) == [("math.pdf", 1), ("math.pdf", 2), ("os/lecture1.pdf", 1)]
    page = db.get_page(conn, PageId("math.pdf", 1))
    assert (page.prompt_version, page.claude_model, page.embed_model) == (
        PARAMS.prompt.version, PARAMS.claude_model, PARAMS.embed_model)


def test_rerun_changes_nothing(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)

    assert plan_sync(conn, Target.resolve(tmp_path), PARAMS).is_empty()
    assert counts(run_sync(conn, tmp_path)) == (0, 0, 0, 2, 0)


def test_rerun_extracts_only_the_changed_page(conn, tmp_path):
    pdf = make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)
    old_hash = db.get_page(conn, PageId("math.pdf", 2)).content_hash

    make_pdf(pdf, ["limits", "integrals"])
    plan = plan_sync(conn, Target.resolve(tmp_path), PARAMS)
    summary = run_sync(conn, tmp_path)

    assert [(j.id.page_num, j.reason) for j in plan.extract] == [(2, ExtractReason.PAGE_CHANGED)]
    assert counts(summary) == (1, 0, 0, 1, 0)
    assert db.get_page(conn, PageId("math.pdf", 2)).content_hash != old_hash


@pytest.mark.parametrize(("change", "reason"), [
    ({"prompt": Prompt("New prompt.")}, ExtractReason.PROMPT_CHANGED),
    ({"claude_model": "mock-claude-2"}, ExtractReason.CLAUDE_MODEL_CHANGED),
])
def test_an_extraction_config_change_extracts_every_page_again(conn, tmp_path, change, reason):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)
    new_params = replace(PARAMS, **change)

    plan = plan_sync(conn, Target.resolve(tmp_path), new_params)
    summary = run_sync(conn, tmp_path, new_params)

    assert [j.reason for j in plan.extract] == [reason, reason]
    assert counts(summary) == (2, 0, 0, 0, 0)
    page = db.get_page(conn, PageId("math.pdf", 1))
    assert (page.prompt_version, page.claude_model) == (
        new_params.prompt.version, new_params.claude_model)
    assert plan_sync(conn, Target.resolve(tmp_path), new_params).is_empty()


def test_a_dpi_change_extracts_every_page_again(conn, tmp_path):
    """dpi isn't stored: the page hashes change instead."""
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)

    plan = plan_sync(conn, Target.resolve(tmp_path), replace(PARAMS, dpi=PARAMS.dpi * 2))

    assert [j.reason for j in plan.extract] == [ExtractReason.PAGE_CHANGED] * 2


def test_an_embedding_model_change_re_embeds_without_extracting(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)
    before = db.get_page(conn, PageId("math.pdf", 1))
    new_params = replace(PARAMS, embed_model="mock-embed-2")

    plan = plan_sync(conn, Target.resolve(tmp_path), new_params)
    summary = apply_plan(conn, plan, new_params, None,
                         MockEmbedder(model_name=new_params.embed_model))

    assert [j.reason for j in plan.embed] == [EmbedReason.EMBED_MODEL_CHANGED] * 2
    assert counts(summary) == (0, 2, 0, 0, 0)
    page = db.get_page(conn, PageId("math.pdf", 1))
    assert page.embed_model == "mock-embed-2"
    assert (page.transcript, page.content_hash) == (before.transcript, before.content_hash)
    assert plan_sync(conn, Target.resolve(tmp_path), new_params).is_empty()


def test_pages_removed_from_a_pdf_are_deleted(conn, tmp_path):
    pdf = make_pdf(tmp_path / "math.pdf", ["limits", "derivatives", "integrals"])
    run_sync(conn, tmp_path)

    make_pdf(pdf, ["limits"])
    summary = run_sync(conn, tmp_path)

    assert counts(summary) == (0, 0, 2, 1, 0)
    assert saved_pages(conn) == [("math.pdf", 1)]


def test_a_deleted_pdf_is_deleted(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])
    old = make_pdf(tmp_path / "os" / "old.pdf", ["old notes", "more old notes"])
    run_sync(conn, tmp_path)

    old.unlink()
    summary = run_sync(conn, tmp_path)

    assert counts(summary) == (0, 0, 2, 1, 0)
    assert saved_pages(conn) == [("math.pdf", 1)]


def test_syncing_a_target_leaves_pages_outside_it(conn, tmp_path):
    math = make_pdf(tmp_path / "math.pdf", ["limits"])
    make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes"])
    run_sync(conn, tmp_path)

    math.unlink()
    summary = run_sync(conn, tmp_path, target=tmp_path / "os")

    assert counts(summary) == (0, 0, 0, 1, 0)
    assert saved_pages(conn) == [("math.pdf", 1), ("os/lecture1.pdf", 1)]


def test_target_limits_sync_to_one_pdf(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])
    target = make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes"])

    run_sync(conn, tmp_path, target=target)

    assert saved_pages(conn) == [("os/lecture1.pdf", 1)]


def test_failed_pages_are_reported_and_retried_next_run(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])

    summary = run_sync(conn, tmp_path, extractor=MockExtractor(fail=True))

    assert counts(summary) == (0, 0, 0, 0, 2)
    assert sorted(f.job.id.page_num for f in summary.failed) == [1, 2]
    assert summary.failed[0].error == "RuntimeError: API down"
    assert saved_pages(conn) == []

    summary = run_sync(conn, tmp_path)
    assert counts(summary) == (2, 0, 0, 0, 0)
