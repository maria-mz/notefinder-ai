from pathlib import Path

import pytest

from notefinder.sync.target import Target
from tests.helpers import make_pdf


def test_resolve_defaults_to_all_of_root(tmp_path):
    assert Target.resolve(tmp_path) == Target(tmp_path, tmp_path)


def test_resolve_makes_paths_absolute(tmp_path, monkeypatch):
    make_pdf(tmp_path / "notes" / "os" / "a.pdf", ["x"])
    monkeypatch.chdir(tmp_path)

    target = Target.resolve(Path("notes/../notes"), Path("notes/os"))

    assert target == Target(tmp_path / "notes", tmp_path / "notes" / "os")


def test_resolve_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="notes folder"):
        Target.resolve(tmp_path / "missing")


def test_resolve_rejects_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError, match="target"):
        Target.resolve(tmp_path, tmp_path / "missing.pdf")


def test_resolve_rejects_target_outside_root(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    outside = make_pdf(tmp_path / "other.pdf", ["x"])
    with pytest.raises(ValueError, match="outside"):
        Target.resolve(root, outside)


def test_resolve_rejects_non_pdf_file(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(ValueError, match="not a PDF"):
        Target.resolve(tmp_path, txt)


def test_scope_of_the_whole_root_is_empty(tmp_path):
    assert Target(tmp_path, tmp_path).scope == ""


def test_scope_of_a_folder_or_pdf_is_its_path_from_root(tmp_path):
    assert Target(tmp_path, tmp_path / "os").scope == "os"
    assert Target(tmp_path, tmp_path / "os" / "lecture3.pdf").scope == "os/lecture3.pdf"


def test_source_file_is_the_pdf_path_from_root(tmp_path):
    target = Target(tmp_path, tmp_path / "os")

    assert target.source_file(tmp_path / "os" / "sub" / "a.pdf") == "os/sub/a.pdf"


def test_pdfs_searches_subfolders_in_sorted_order(tmp_path):
    b = make_pdf(tmp_path / "os" / "b.pdf", ["x"])
    a = make_pdf(tmp_path / "a.pdf", ["x"])
    (tmp_path / "readme.txt").write_text("not a pdf")

    assert Target(tmp_path, tmp_path).pdfs() == [a, b]


def test_pdfs_of_a_folder_skips_pdfs_outside_it(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["x"])
    b = make_pdf(tmp_path / "os" / "b.pdf", ["x"])

    assert Target(tmp_path, tmp_path / "os").pdfs() == [b]


def test_pdfs_of_a_pdf_is_just_that_pdf(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", ["x"])
    make_pdf(tmp_path / "b.pdf", ["x"])

    assert Target(tmp_path, a).pdfs() == [a]
