import pytest
from typer.testing import CliRunner

from notefinder import cli
from notefinder import db
from notefinder.config import AppConfig
from notefinder.db import PageId
from notefinder.prompt import Prompt
from notefinder.sync.plan import SyncParams
from tests.helpers import make_pdf, save_test_page

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Ensures the tests exit before connecting to the database, unless they use cli_db."""
    def connect():
        raise AssertionError("cli.connect was called")
    monkeypatch.setattr(cli, "connect", connect)


@pytest.fixture
def cli_db(conn, test_db_url, monkeypatch):
    """Points the CLI at the test DB. Yields a separate connection for checking results."""
    monkeypatch.setattr(cli, "connect", lambda: db.connect(test_db_url))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return conn


def test_sync_params_takes_its_settings_from_the_app_config(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Transcribe this page.")
    app_config = AppConfig(prompt_path=prompt, claude_model="claude-x",
                           embed_model="embed-x", dpi=99, workers=3)

    assert cli.sync_params(app_config, num_workers=7) == SyncParams(
        prompt=Prompt("Transcribe this page."),
        claude_model="claude-x",
        embed_model="embed-x",
        dpi=99,
        num_workers=7,  # from --workers, not app_config.workers
    )


def saved_pages(conn) -> list[tuple[str, int]]:
    return conn.execute("SELECT source_file, page_num FROM pages ORDER BY page_num").fetchall()


def test_sync_rejects_a_target_outside_the_notes_dir(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    outside = tmp_path / "other.pdf"
    outside.write_bytes(b"")

    result = runner.invoke(cli.app, ["sync", str(outside), "--notes-dir", str(notes)])

    assert result.exit_code == 2
    assert "outside" in result.output


def test_sync_needs_an_api_key_to_extract(cli_db, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])

    result = runner.invoke(cli.app, ["sync", "--notes-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
    assert saved_pages(cli_db) == []


def test_dry_run_shows_the_plan_and_changes_nothing(cli_db, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    save_test_page(cli_db, PageId("gone.pdf", 1))

    result = runner.invoke(cli.app, ["sync", "--notes-dir", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert "2  (2 new)" in result.output
    assert "gone.pdf p.1" in result.output
    assert saved_pages(cli_db) == [("gone.pdf", 1)]


def test_sync_asks_before_deleting(cli_db, tmp_path):
    save_test_page(cli_db, PageId("gone.pdf", 1))

    result = runner.invoke(cli.app, ["sync", "--notes-dir", str(tmp_path)], input="n\n")

    assert result.exit_code == 1
    assert "Delete 1 pages" in result.output
    assert saved_pages(cli_db) == [("gone.pdf", 1)]


@pytest.mark.parametrize(("args", "input"), [
    pytest.param(["--yes"], None, id="--yes"),
    pytest.param([], "y\n", id="confirmed"),
])
def test_sync_deletes_pages_that_are_gone_without_an_api_key(cli_db, tmp_path, args, input):
    save_test_page(cli_db, PageId("gone.pdf", 1))

    result = runner.invoke(cli.app, ["sync", "--notes-dir", str(tmp_path), *args], input=input)

    assert result.exit_code == 0, result.output
    assert "1 deleted" in result.output
    assert saved_pages(cli_db) == []
