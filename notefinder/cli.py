"""The command line: ``sync``, ``search``, and ``eval``."""
import logging
import os
import shutil
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable

import psycopg
import typer
import yaml

from notefinder import db
from notefinder.config import APP_CONFIG, AppConfig
from notefinder.embed import Embedder
from notefinder.evaluate import (
    LabelError, QueryResult, SearchMethod, check_labels, evaluate, load_labels,
    run_record, save_run,
)
from notefinder.extract import Extractor, make_client
from notefinder.prompt import Prompt
from notefinder.search import Bm25Index, search as run_search
from notefinder.sync.apply import SyncSummary, apply_plan
from notefinder.sync.plan import Job, Plan, SyncParams, plan_sync
from notefinder.sync.target import Target

NotesDir = Annotated[Path, typer.Option(
    envvar="NOTEFINDER_NOTES_DIR",
    help="Root folder of the notes. Keep it the same between runs: "
         "pages are stored by their path relative to it.",
)]

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Search handwritten notes by topic."""
    # only warnings from libraries, but info from us
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("notefinder").setLevel(logging.INFO)


def fail(message: str, code: int = 1) -> typer.Exit:
    """Print an error, and return the exit for the caller to raise.

    :param message: What went wrong.
    :param code: The exit code: 2 for bad input, 1 otherwise.
    :returns: The exit to raise.
    """
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def connect() -> psycopg.Connection:
    """Connect to the database at ``NOTEFINDER_DB_URL``, or the configured default.

    :returns: The connection.
    :raises typer.Exit: If the database can't be reached.
    """
    try:
        return db.connect(os.environ.get("NOTEFINDER_DB_URL", APP_CONFIG.db_url))
    except psycopg.OperationalError as e:
        raise fail(f"could not connect to the database (is Docker running?)\n{e}")


def sync_params(app_config: AppConfig, num_workers: int) -> SyncParams:
    """Gather the sync settings from the app config and the ``--workers`` option.

    :param app_config: The app's config to provide sync params.
    :param num_workers: How many pages to process at once.
    :returns: The settings to sync with.
    :raises typer.Exit: If the prompt file can't be read.
    """
    try:
        prompt = Prompt.from_file(app_config.prompt_path)
    except OSError as e:
        raise fail(f"could not read the prompt: {e}")

    return SyncParams(
        prompt=prompt,
        claude_model=app_config.claude_model,
        embed_model=app_config.embed_model,
        dpi=app_config.dpi,
        num_workers=num_workers,
    )


def page_ranges(page_nums: list[int]) -> str:
    """Format page numbers, collapsing consecutive runs.

    :param page_nums: The page numbers, in any order.
    :returns: E.g. ``"pp. 1-3, 7"`` for ``[1, 2, 3, 7]``, or ``"p. 4"`` for ``[4]``.
    """
    runs: list[list[int]] = []
    for n in sorted(page_nums):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    return ("p. " if len(page_nums) == 1 else "pp. ") + text


def format_duration(seconds: float) -> str:
    """Format a duration for display.

    :param seconds: The duration.
    :returns: E.g. ``"4.52s"``, or ``"3m 12s"`` from a minute up.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}m {seconds}s"


def print_plan(plan: Plan, params: SyncParams, *, detailed: bool = False) -> None:
    """Print what a sync would do.

    :param plan: The plan to print.
    :param params: The settings it was made with.
    :param detailed: Also list the files and pages of each extract and re-embed job.
    """
    def line(action: str, jobs: list[Job]) -> None:
        reasons = Counter(job.reason for job in jobs).most_common()
        detail = ", ".join(f"{n} {reason}" for reason, n in reasons)
        typer.echo(f"  {action:<10} {len(jobs):>4}" + (f"  ({detail})" if detail else ""))

        if not detailed:
            return
        groups: dict[tuple[str, str], list[int]] = {}
        for job in jobs:
            key = (job.id.source_file, job.reason)
            groups.setdefault(key, []).append(job.id.page_num)
        for (source_file, reason), page_nums in sorted(groups.items()):
            typer.echo(f"    {source_file}  {page_ranges(page_nums)}  ({reason})")

    typer.echo(
        f"prompt {params.prompt.version}, {params.claude_model}, {params.embed_model}"
    )
    line("extract", plan.extract)
    line("re-embed", plan.embed)
    typer.echo(f"  {'delete':<10} {len(plan.delete):>4}")
    for page_id in plan.delete:
        typer.echo(f"    {page_id}")
    typer.echo(f"  {'unchanged':<10} {plan.unchanged:>4}")


def print_sync_summary(summary: SyncSummary, seconds: float) -> None:
    """Print what a sync did, then each failure to stderr.

    :param summary: What the sync did.
    :param seconds: How long it took.
    """
    typer.echo(f"\n{summary.extracted} extracted, {summary.embedded} re-embedded, "
               f"{summary.deleted} deleted, {len(summary.failed)} failed "
               f"in {format_duration(seconds)}")
    for failure in summary.failed:
        typer.echo(f"  {failure.job.id}: {failure.error}", err=True)


@app.command()
def sync(
    notes_dir: NotesDir,
    target_path: Annotated[Path | None, typer.Argument(
        metavar="TARGET",
        help="A PDF or subfolder of the notes to sync. Default: all of them.",
    )] = None,
    workers: Annotated[int, typer.Option(min=1)] = APP_CONFIG.workers,
    dry_run: Annotated[bool, typer.Option(
        help="Show what would change, and change nothing.",
    )] = False,
    yes: Annotated[bool, typer.Option(
        "--yes", "-y", help="Delete pages that are gone without asking.",
    )] = False,
) -> None:
    """Bring the database in line with the notes, the prompt, and the models."""
    # cheap checks first, before rendering any pages
    try:
        target = Target.resolve(notes_dir, target_path)
    except (FileNotFoundError, ValueError) as e:
        raise fail(str(e), code=2)

    params = sync_params(APP_CONFIG, workers)

    with connect() as conn:
        plan = plan_sync(conn, target, params)
        print_plan(plan, params, detailed=dry_run)

        if dry_run or plan.is_empty():
            return
        if plan.extract and not os.environ.get("ANTHROPIC_API_KEY"):
            raise fail("ANTHROPIC_API_KEY is not set")
        if plan.delete and not yes:
            typer.confirm(
                f"Delete {len(plan.delete)} pages from the database?", abort=True
            )

        start = time.perf_counter()  # after the prompt, so waiting on it doesn't count

        extractor = (
            Extractor(
                make_client(APP_CONFIG.api_max_retries),
                params.claude_model,
                APP_CONFIG.max_tokens,
            )
            if plan.extract else None
        )
        embedder = (
            Embedder(params.embed_model, APP_CONFIG.query_prefix)
            if plan.jobs else None
        )

        summary = apply_plan(conn, plan, params, extractor, embedder)

    print_sync_summary(summary, time.perf_counter() - start)

    if summary.failed:
        raise typer.Exit(1)


def print_search_results(results: list[db.SearchResult]) -> None:
    """Print each result's rank, score, and page, then its summary, shortened.

    :param results: The results, best first.
    """
    width = min(shutil.get_terminal_size().columns, 100)

    def indented(text: str) -> str:
        return textwrap.fill(
            text, width=width, initial_indent="   ", subsequent_indent="   "
        )

    for rank, r in enumerate(results, start=1):
        typer.echo(f"{rank}. {r.score:.3f}  {r.id}")
        typer.echo(indented(textwrap.shorten(r.summary, width=160)))
        typer.echo()


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="A topic, e.g. 'cache coherence'.")],
    k: Annotated[int, typer.Option(
        "-k", min=1, help="How many pages to show.",
    )] = APP_CONFIG.search_k,
) -> None:
    """Find the pages most related to a topic."""
    if not query.strip():
        raise fail("query is empty", code=2)

    with connect() as conn:
        embedder = Embedder(APP_CONFIG.embed_model, APP_CONFIG.query_prefix)
        results = run_search(conn, embedder, query, k)

    if not results:
        typer.echo(
            f"No pages embedded with {APP_CONFIG.embed_model} yet. Run `sync` first."
        )
        return

    print_search_results(results)


def make_rank(
    method: SearchMethod,
    conn,
    stored: list[db.StoredPage],
    params: SyncParams,
) -> Callable[[str], list[db.PageId]]:
    """Build the ranking function an eval scores.

    :param method: What ranks the pages.
    :param conn: The database connection. Must stay open while the function is used.
    :param stored: Every stored page, already checked to be current.
    :param params: Supplies the embedding model.
    :returns: A function from a query to pages, best first.
    """
    if method == SearchMethod.BM25:
        return Bm25Index(db.get_transcripts(conn)).rank
    if method == SearchMethod.BM25_EMBEDDED_TEXT:
        return Bm25Index({s.id: s.embedded_text for s in stored}).rank

    embedder = Embedder(params.embed_model, APP_CONFIG.query_prefix)

    # rank every page, so a miss still shows where it landed
    def rank(query: str) -> list[db.PageId]:
        return [r.id for r in run_search(conn, embedder, query, len(stored))]
    return rank


def print_eval_results(results: list[QueryResult], k: int) -> None:
    """Print each query's recall, and the relevant pages it missed with their ranks.

    :param results: One result per query.
    :param k: The k the results were scored at.
    """
    typer.echo(f"recall@{k}  query")
    for r in results:
        typer.echo(f"  {r.recall:.2f}    {r.query}")
        for page_id, page_rank in r.missed(k):
            typer.echo(f"            missed: {page_id} (rank {page_rank or 'none'})")


def print_eval_summary(record: dict, run_path: Path) -> None:
    """Print a run's mean recall and settings, and where it was saved.

    :param record: The run's record, from :func:`~notefinder.evaluate.run_record`.
    :param run_path: Where the record was saved.
    """
    r = record
    typer.echo(f"\nmean recall@{r['k']} {r['mean_recall']:.2f}  "
               f"({r['num_queries']} queries, {r['num_pages']} pages)")
    typer.echo(f"{r['method']} · prompt {r['prompt_version']} · {r['claude_model']} · "
               f"{r['embed_model']}")
    typer.echo(f"saved {run_path}")


@app.command("eval")
def eval_(
    notes_dir: NotesDir,
    labels_path: Annotated[Path, typer.Option(
        "--labels", help="The hand-labelled queries.",
    )] = APP_CONFIG.labels_path,
    k: Annotated[int, typer.Option(
        "-k", min=1, help="Score the top k results of each query.",
    )] = APP_CONFIG.eval_k,
    method: Annotated[SearchMethod, typer.Option(
        help="Rank by embeddings (as `search` does). Or, as baselines, by keywords "
             "in the transcripts, or in the summary and concepts the embeddings "
             "are made from.",
    )] = SearchMethod.EMBEDDING,
) -> None:
    """Score search against the labelled queries (mean recall@k), and save the run."""
    # cheap checks first
    try:
        queries = load_labels(labels_path)
    except (OSError, yaml.YAMLError, LabelError) as e:
        raise fail(f"bad labels file {labels_path}: {e}", code=2)
    try:
        target = Target.resolve(notes_dir)
    except (FileNotFoundError, ValueError) as e:
        raise fail(str(e), code=2)

    params = sync_params(APP_CONFIG, APP_CONFIG.workers)

    with connect() as conn:
        # the labels describe the notes, so the database must match them exactly
        if not plan_sync(conn, target, params).is_empty():
            raise fail("the database doesn't match the notes. Run `sync` first.")

        stored = db.get_stored_pages(conn)
        if problem := check_labels(queries, {s.id for s in stored}):
            raise fail(f"can't evaluate: {problem}")

        results = evaluate(queries, make_rank(method, conn, stored, params), k)

    print_eval_results(results, k)

    now = datetime.now(timezone.utc)
    record = run_record(
        results, k=k, method=method, params=params, num_pages=len(stored),
        timestamp=now.isoformat(timespec="seconds"),
    )
    run_path = save_run(record, APP_CONFIG.runs_dir, now)

    print_eval_summary(record, run_path)
