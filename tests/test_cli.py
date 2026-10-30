import pytest
from typer.testing import CliRunner

import cli
from ingest import load_prompt

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Ensures the tests exit before connecting to the database or loading the model."""
    def connect(*args, **kwargs):
        raise AssertionError("db.connect was called")
    monkeypatch.setattr(cli.db, "connect", connect)


def test_load_prompt_version_changes_only_when_the_text_does(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Transcribe this page.")
    text, version = load_prompt(path)
    assert text == "Transcribe this page."
    assert load_prompt(path) == (text, version)

    path.write_text("Transcribe this page carefully.")
    assert load_prompt(path)[1] != version


def test_ingest_rejects_a_target_outside_the_notes_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    notes = tmp_path / "notes"
    notes.mkdir()
    outside = tmp_path / "other.pdf"
    outside.write_bytes(b"")

    result = runner.invoke(cli.app, ["ingest", str(outside), "--notes-dir", str(notes)])

    assert result.exit_code == 2
    assert "outside" in result.output


def test_ingest_requires_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["ingest", "--notes-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
