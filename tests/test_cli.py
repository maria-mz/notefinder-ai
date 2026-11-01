import json
from dataclasses import replace

import psycopg
import pytest
from typer.testing import CliRunner

from notefinder import cli
from notefinder import db
from notefinder.config import AppConfig
from notefinder.db import PageId
from notefinder.evaluate import QueryResult
from notefinder.prompt import Prompt
from notefinder.render import hash_image, render_page
from notefinder.sync.apply import JobFailure, SyncSummary
from notefinder.sync.plan import ExtractJob, ExtractReason, SyncParams, embedding_text
from tests.helpers import (
    PARAMS, MockEmbedder, axis_vector, make_pdf, save_test_page, saved_pages,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Ensures tests exit before connecting to the database, unless they use cli_db."""
    def connect():
        raise AssertionError("cli.connect was called")
    monkeypatch.setattr(cli, "connect", connect)


@pytest.fixture
def cli_db(conn, test_db_url, monkeypatch):
    """Points the CLI at the test DB. Returns a connection for checking results."""
    monkeypatch.setattr(cli, "connect", lambda: db.connect(test_db_url))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return conn


@pytest.fixture
def eval_setup(cli_db, tmp_path, monkeypatch):
    """Runs eval with PARAMS, a mock embedder, and runs saved under tmp_path."""
    monkeypatch.setattr(cli, "sync_params", lambda app_config, num_workers: PARAMS)
    monkeypatch.setattr(cli, "Embedder", lambda model_name, query_prefix: MockEmbedder())
    app_config = replace(cli.APP_CONFIG, runs_dir=tmp_path / "runs")
    monkeypatch.setattr(cli, "APP_CONFIG", app_config)
    return cli_db


def run_sync_command(notes_dir, *args, input=None):
    return runner.invoke(cli.app, ["sync", "--notes-dir", str(notes_dir), *args],
                         input=input)


def run_eval_command(notes_dir, labels, *args):
    return runner.invoke(cli.app, ["eval", "--notes-dir", str(notes_dir),
                                   "--labels", str(labels), *args])


def run_shell_command(*queries):
    """Run the shell, typing each query on its own line, then quitting with Ctrl-D."""
    return runner.invoke(cli.app, ["shell"], input="".join(q + "\n" for q in queries))


def save_notes(conn, notes_dir, *pages: dict) -> None:
    """Write notes.pdf with one page per dict, and store each page in sync with it.

    Each dict overrides save_test_page's fields for that page.
    """
    texts = [f"page {n}" for n in range(1, len(pages) + 1)]
    pdf = make_pdf(notes_dir / "notes.pdf", texts)
    for n, overrides in enumerate(pages, start=1):
        content_hash = hash_image(render_page(pdf, n, PARAMS.dpi))
        save_test_page(conn, PageId("notes.pdf", n),
                       **({"content_hash": content_hash} | overrides))


def write_labels(path, *queries: tuple[str, list[int]]):
    lines = ["queries:"]
    for query, pages in queries:
        lines.append(f"  - query: {query}")
        lines.append("    relevant:")
        lines += [f"      - {{file: notes.pdf, page: {p}}}" for p in pages]
    path.write_text("\n".join(lines) + "\n")
    return path


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


def test_sync_rejects_a_target_outside_the_notes_dir(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    outside = tmp_path / "other.pdf"
    outside.write_bytes(b"")

    result = run_sync_command(notes, str(outside))

    assert result.exit_code == 2
    assert "outside" in result.output


def test_sync_needs_an_api_key_to_extract(cli_db, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits"])

    result = run_sync_command(tmp_path)

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
    assert saved_pages(cli_db) == []


def test_dry_run_shows_the_plan_and_changes_nothing(cli_db, tmp_path):
    make_pdf(tmp_path / "math.pdf", ["limits", "derivatives"])
    save_test_page(cli_db, PageId("gone.pdf", 1))

    result = run_sync_command(tmp_path, "--dry-run")

    assert result.exit_code == 0
    assert "2  (2 new)" in result.output
    assert "gone.pdf p.1" in result.output
    assert "math.pdf  pp. 1-2  (new)" in result.output
    assert saved_pages(cli_db) == [("gone.pdf", 1)]


def test_sync_asks_before_deleting(cli_db, tmp_path):
    save_test_page(cli_db, PageId("gone.pdf", 1))

    result = run_sync_command(tmp_path, input="n\n")

    assert result.exit_code == 1
    assert "Delete 1 pages" in result.output
    assert saved_pages(cli_db) == [("gone.pdf", 1)]


@pytest.mark.parametrize(("args", "input"), [
    pytest.param(["--yes"], None, id="--yes"),
    pytest.param([], "y\n", id="confirmed"),
])
def test_sync_deletes_pages_that_are_gone_without_an_api_key(cli_db, tmp_path, args,
                                                             input):
    # A sync that only deletes doesn't need an API key.
    save_test_page(cli_db, PageId("gone.pdf", 1))

    result = run_sync_command(tmp_path, *args, input=input)

    assert result.exit_code == 0, result.output
    assert "1 deleted" in result.output
    assert saved_pages(cli_db) == []


def test_search_rejects_an_empty_query():
    result = runner.invoke(cli.app, ["search", "  "])

    assert result.exit_code == 2
    assert "query is empty" in result.output


def test_search_says_to_sync_when_nothing_is_embedded(cli_db, monkeypatch):
    monkeypatch.setattr(cli, "Embedder", lambda model_name, query_prefix:
                        MockEmbedder(model_name=model_name))

    result = runner.invoke(cli.app, ["search", "heaps"])

    assert result.exit_code == 0, result.output
    assert "Run `sync` first" in result.output


@pytest.fixture
def shell_embedder(cli_db, monkeypatch):
    """Has the shell embed every query as axis_vector(1, 0). Returns each model load."""
    loads = []
    def load(model_name, query_prefix):
        loads.append(model_name)
        return MockEmbedder(lambda query: axis_vector(1, 0), model_name=model_name)
    monkeypatch.setattr(cli, "Embedder", load)
    return loads


def test_shell_searches_each_query_and_loads_the_model_once(cli_db, shell_embedder):
    save_test_page(cli_db, PageId("heaps.pdf", 1),
                   embed_model=cli.APP_CONFIG.embed_model, embedding=axis_vector(1, 0))

    result = run_shell_command("heaps", "", "tries")

    assert result.exit_code == 0, result.output
    assert result.output.count("heaps.pdf  p.1") == 2
    assert len(shell_embedder) == 1


def test_shell_says_to_sync_when_nothing_is_embedded(shell_embedder):
    result = run_shell_command("heaps")

    assert result.exit_code == 0, result.output
    assert "Run `sync` first" in result.output


@pytest.mark.parametrize(("page_nums", "expected"), [
    ([3], "p. 3"),
    ([1, 2, 3], "pp. 1-3"),
    ([7, 1, 2, 3, 5], "pp. 1-3, 5, 7"),
])
def test_page_ranges_collapses_consecutive_pages(page_nums, expected):
    assert cli.page_ranges(page_nums) == expected


@pytest.mark.parametrize(("seconds", "expected"), [
    (4.519, "4.52s"),
    (59.99, "59.99s"),
    (60, "1m 0s"),
    (192.4, "3m 12s"),
])
def test_format_duration(seconds, expected):
    assert cli.format_duration(seconds) == expected


def test_print_sync_summary_prints_counts_and_failures_to_stderr(capsys, tmp_path):
    job = ExtractJob(PageId("os.pdf", 2), tmp_path / "os.pdf", ExtractReason.NEW)
    summary = SyncSummary(extracted=3, embedded=1, deleted=0, unchanged=5,
                          failed=[JobFailure(job, "RuntimeError: API down")])

    cli.print_sync_summary(summary, 4.519)

    out, err = capsys.readouterr()
    assert out == "\n3 extracted, 1 re-embedded, 0 deleted, 1 failed in 4.52s\n"
    assert err == f"  {job.id}: RuntimeError: API down\n"


def test_print_search_results_prints_each_page_summary_and_concepts(capsys,
                                                                    monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    summary = ("OS notes: " + "paging and TLBs, " * 12).rstrip(", ") + "."  # wraps
    results = [
        db.SearchResult(PageId("os/memory.pdf", 2), summary, ["TLB", "paging"], 0.8123),
        db.SearchResult(PageId("caches.pdf", 5), "Cache coherence.", ["MESI"], 0.7),
    ]

    cli.print_search_results(results, seconds=0.0149)

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith(" 1  memory.pdf  p.2")
    assert lines[0].endswith("os/ · 0.81")
    concepts = lines.index("    ▸ TLB · paging")
    assert " ".join(line.strip() for line in lines[1:concepts]) == summary  # in full
    assert lines[concepts + 2].startswith(" 2  caches.pdf  p.5")
    assert lines[concepts + 2].endswith(" 0.70")
    assert lines[-1] == "2 pages · 15 ms"
    assert all(line == line.rstrip() for line in lines)


def test_concepts_text_fits_the_width():
    concepts = ["paging", "translation lookaside buffer", "segmentation"]

    assert cli.concepts_text(concepts, 60).plain == (
        "▸ paging · translation lookaside buffer · segmentation")
    assert cli.concepts_text(concepts, 30).plain == "▸ paging · …"
    assert cli.concepts_text(concepts, 5).plain == "▸ paging · …"  # always one


def test_print_eval_results_prints_recall_and_misses(capsys):
    results = [
        QueryResult("heaps", recall=1.0, ranks={PageId("ds.pdf", 1): 1}),
        QueryResult("tries", recall=0.0,
                    ranks={PageId("ds.pdf", 3): 7, PageId("ds.pdf", 2): None}),
    ]

    cli.print_eval_results(results, k=5)

    assert capsys.readouterr().out.splitlines() == [
        "recall@5  query",
        "  1.00    heaps",
        "  0.00    tries",
        "            missed: ds.pdf p.2 (rank none)",
        "            missed: ds.pdf p.3 (rank 7)",
    ]


def test_print_eval_summary_prints_from_the_record(capsys, tmp_path):
    record = {"k": 5, "mean_recall": 0.8333, "num_queries": 30, "num_pages": 100,
              "method": "embedding", "prompt_version": "abc123",
              "claude_model": "claude-x", "embed_model": "embed-y"}

    cli.print_eval_summary(record, tmp_path / "run.json")

    assert capsys.readouterr().out.splitlines() == [
        "",
        "mean recall@5 0.83  (30 queries, 100 pages)",
        "embedding · prompt abc123 · claude-x · embed-y",
        f"saved {tmp_path / 'run.json'}",
    ]


def test_eval_rejects_bad_labels_before_connecting(tmp_path):
    labels = tmp_path / "labels.yaml"
    labels.write_text("queries: []\nnot: [valid")

    result = run_eval_command(tmp_path, labels)

    assert result.exit_code == 2
    assert "bad labels file" in result.output


@pytest.mark.parametrize("overrides", [
    pytest.param({"prompt_version": "old"}, id="settings"),
    pytest.param({"content_hash": "old"}, id="notes"),
])
def test_eval_refuses_a_database_that_doesnt_match_the_notes(eval_setup, tmp_path,
                                                             overrides):
    save_notes(eval_setup, tmp_path / "notes", {}, overrides)
    labels = write_labels(tmp_path / "labels.yaml", ("near", [1]))

    result = run_eval_command(tmp_path / "notes", labels)

    assert result.exit_code == 1
    assert "Run `sync` first" in result.output
    assert not (tmp_path / "runs").exists()


def test_eval_refuses_labelled_pages_missing_from_the_database(eval_setup, tmp_path):
    save_notes(eval_setup, tmp_path / "notes", {})
    labels = write_labels(tmp_path / "labels.yaml", ("near", [1, 9]))

    result = run_eval_command(tmp_path / "notes", labels)

    assert result.exit_code == 1
    assert "1 labelled pages aren't in the database: notes.pdf p.9" in result.output


def test_eval_scores_and_saves_the_run(eval_setup, tmp_path):
    save_notes(eval_setup, tmp_path / "notes", {})
    labels = write_labels(tmp_path / "labels.yaml", ("heaps", [1]))

    result = run_eval_command(tmp_path / "notes", labels, "-k", "1")

    assert result.exit_code == 0, result.output
    assert "mean recall@1 1.00" in result.output
    [run_file] = (tmp_path / "runs").iterdir()
    record = json.loads(run_file.read_text())
    assert (record["method"], record["mean_recall"], record["num_pages"]) == (
        "embedding", 1.0, 1)


# BM25 gives a word on half the pages or more no weight, so these tests use three
# pages with the query word on one


def test_eval_bm25_ranks_by_transcript(eval_setup, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "Embedder", lambda *args: pytest.fail("loaded the embedder"))
    save_notes(eval_setup, tmp_path / "notes",
               {"transcript": "heaps"}, {"transcript": "graphs"}, {"transcript": "tries"})
    labels = write_labels(tmp_path / "labels.yaml", ("heaps", [1]))

    result = run_eval_command(tmp_path / "notes", labels, "-k", "1", "--method", "bm25")

    assert result.exit_code == 0, result.output
    assert "mean recall@1 1.00" in result.output


def test_eval_bm25_embedded_text_ranks_by_summary_and_concepts(eval_setup, tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(cli, "Embedder", lambda *args: pytest.fail("loaded the embedder"))
    save_notes(eval_setup, tmp_path / "notes", *(
        {"concepts": [topic], "embedded_text": embedding_text("Summary", [topic])}
        for topic in ["heaps", "graphs", "tries"]
    ))
    labels = write_labels(tmp_path / "labels.yaml", ("heaps", [1]))

    result = run_eval_command(tmp_path / "notes", labels, "-k", "1",
                              "--method", "bm25-embedded-text")

    assert result.exit_code == 0, result.output
    assert "mean recall@1 1.00" in result.output
