from dataclasses import replace

import pytest

from notefinder.sync.apply import apply_plan
from notefinder.sync.plan import plan_sync
from notefinder.prompt import Prompt
from notefinder.sync.target import Target
from tests.helpers import PARAMS, MockEmbedder, MockExtractor, make_pdf, saved_pages


def sync_plan(conn, root, params=PARAMS, target=None):
    return plan_sync(conn, Target.resolve(root, target), params)


def run_sync(conn, root, params=PARAMS, extractor=None, target=None):
    plan = sync_plan(conn, root, params, target)
    return apply_plan(conn, plan, params, extractor or MockExtractor(),
                      MockEmbedder(model_name=params.embed_model))


def check_summary(summary, *, extracted: int, embedded: int, deleted: int,
                  unchanged: int, failed: int) -> None:
    """Check every count in a sync summary."""
    assert summary.extracted == extracted
    assert summary.embedded == embedded
    assert summary.deleted == deleted
    assert summary.unchanged == unchanged
    assert len(summary.failed) == failed


def test_rerun_changes_nothing(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)

    assert sync_plan(conn, tmp_path).is_empty()
    summary = run_sync(conn, tmp_path)
    check_summary(summary, extracted=0, embedded=0, deleted=0, unchanged=2, failed=0)


@pytest.mark.parametrize(("change", "extracted", "embedded"), [
    ({"prompt": Prompt("New prompt.")}, 2, 0),
    ({"claude_model": "mock-claude-2"}, 2, 0),
    ({"dpi": PARAMS.dpi * 2}, 2, 0),
    ({"embed_model": "mock-embed-2"}, 0, 2),
])
def test_a_sync_param_change_updates_every_page(conn, tmp_path, change, extracted,
                                                embedded):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_sync(conn, tmp_path)
    new_params = replace(PARAMS, **change)

    summary = run_sync(conn, tmp_path, new_params)

    check_summary(summary, extracted=extracted, embedded=embedded, deleted=0,
                  unchanged=0, failed=0)
    assert sync_plan(conn, tmp_path, new_params).is_empty()


def test_a_deleted_pdf_is_deleted(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])
    old = make_pdf(tmp_path / "os" / "old.pdf", ["old notes", "more old notes"])
    run_sync(conn, tmp_path)

    old.unlink()
    summary = run_sync(conn, tmp_path)

    check_summary(summary, extracted=0, embedded=0, deleted=2, unchanged=1, failed=0)
    assert saved_pages(conn) == [("math.pdf", 1)]


def test_syncing_a_target_leaves_pages_outside_it(conn, tmp_path):
    math = make_pdf(tmp_path / "math.pdf", ["limits"])
    make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes"])
    run_sync(conn, tmp_path)

    math.unlink()
    summary = run_sync(conn, tmp_path, target=tmp_path / "os")

    check_summary(summary, extracted=0, embedded=0, deleted=0, unchanged=1, failed=0)
    assert saved_pages(conn) == [("math.pdf", 1), ("os/lecture1.pdf", 1)]


def test_failed_pages_are_reported_and_retried_next_run(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])

    summary = run_sync(conn, tmp_path, extractor=MockExtractor(fail=True))

    check_summary(summary, extracted=0, embedded=0, deleted=0, unchanged=0, failed=2)
    assert sorted(f.job.id.page_num for f in summary.failed) == [1, 2]
    assert summary.failed[0].error == "RuntimeError: API down"
    assert saved_pages(conn) == []

    summary = run_sync(conn, tmp_path)
    check_summary(summary, extracted=2, embedded=0, deleted=0, unchanged=0, failed=0)
