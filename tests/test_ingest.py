from pathlib import Path

import pymupdf
import pytest

import db
from ingest import find_pdfs, ingest, validate_target
from tests.helpers import MockEmbedder, MockExtractor

PROMPT = "Transcribe this page."
VERSION = "v1"


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    """A PDF with one page per text, so pages render (and hash) differently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        for text in page_texts:
            doc.new_page().insert_text((72, 72), text)
        doc.save(path)
    return path


def run_ingest(conn, root, extractor=None, target=None):
    return ingest(conn, root, extractor or MockExtractor(), MockEmbedder(),
                  prompt=PROMPT, prompt_version=VERSION, num_workers=2, target=target)


def saved_pages(conn) -> list[tuple[str, int]]:
    return conn.execute(
        "SELECT source_file, page_num FROM pages ORDER BY source_file, page_num"
    ).fetchall()


def test_validate_target_defaults_to_root(tmp_path):
    assert validate_target(tmp_path) == (tmp_path, tmp_path)


def test_validate_target_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_target(tmp_path / "missing")


def test_validate_target_rejects_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_target(tmp_path, tmp_path / "missing.pdf")


def test_validate_target_rejects_target_outside_root(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    outside = make_pdf(tmp_path / "other.pdf", ["x"])
    with pytest.raises(ValueError, match="outside"):
        validate_target(root, outside)


def test_validate_target_rejects_non_pdf_file(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(ValueError, match="not a PDF"):
        validate_target(tmp_path, txt)


def test_find_pdfs_searches_subfolders_in_sorted_order(tmp_path):
    b = make_pdf(tmp_path / "os" / "b.pdf", ["x"])
    a = make_pdf(tmp_path / "a.pdf", ["x"])
    (tmp_path / "readme.txt").write_text("not a pdf")
    assert find_pdfs(tmp_path) == [a, b]


def test_ingest_saves_every_page(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes"])

    summary = run_ingest(conn, tmp_path)

    assert (summary.unchanged, summary.saved, summary.failed) == (0, 3, [])
    assert saved_pages(conn) == [("math.pdf", 1), ("math.pdf", 2), ("os/lecture1.pdf", 1)]
    assert db.get_page(conn, "math.pdf", 1).prompt_version == VERSION


def test_rerun_skips_unchanged_pages(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_ingest(conn, tmp_path)

    summary = run_ingest(conn, tmp_path)

    assert (summary.unchanged, summary.saved, summary.failed) == (2, 0, [])


def test_rerun_processes_only_the_changed_page(conn, tmp_path):
    pdf = make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    run_ingest(conn, tmp_path)
    old_hash = db.get_page(conn, "math.pdf", 2).content_hash

    make_pdf(pdf, ["limits", "integrals"])
    summary = run_ingest(conn, tmp_path)

    assert (summary.unchanged, summary.saved) == (1, 1)
    assert db.get_page(conn, "math.pdf", 2).content_hash != old_hash


def test_target_limits_ingest_to_one_pdf(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])
    target = make_pdf(tmp_path / "os" / "lecture1.pdf", ["processes"])

    run_ingest(conn, tmp_path, target=target)

    assert saved_pages(conn) == [("os/lecture1.pdf", 1)]


def test_failed_pages_are_reported_and_retried_next_run(conn, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])

    summary = run_ingest(conn, tmp_path, MockExtractor(fail=True))

    assert summary.saved == 0
    assert sorted(f.job.page_num for f in summary.failed) == [1, 2]
    assert summary.failed[0].error == "RuntimeError: API down"
    assert saved_pages(conn) == []

    summary = run_ingest(conn, tmp_path)
    assert (summary.unchanged, summary.saved, summary.failed) == (0, 2, [])
