import logging
import os
import shutil
import textwrap
from collections import Counter
from pathlib import Path
from typing import Annotated

import psycopg
import typer

from notefinder import db
from notefinder.config import APP_CONFIG, AppConfig
from notefinder.embeddings import Embedder
from notefinder.extract import Extractor, make_client
from notefinder.prompt import Prompt
from notefinder.search import search as run_search
from notefinder.sync.apply import apply_plan
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # logs every API request
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)  # logs model loading


def fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def connect() -> psycopg.Connection:
    try:
        return db.connect(os.environ.get("NOTEFINDER_DB_URL", APP_CONFIG.db_url))
    except psycopg.OperationalError as e:
        raise fail(f"could not connect to the database (is Docker running?)\n{e}")


def sync_params(app_config: AppConfig, num_workers: int) -> SyncParams:
    """The sync settings, from the app config and the --workers option.

    :param app_config: Supplies every setting but the number of workers.
    :param num_workers: How many pages to process at once.
    :returns: The settings to sync with. Reads the prompt file.
    """
    return SyncParams(
        prompt=Prompt.from_file(app_config.prompt_path),
        claude_model=app_config.claude_model,
        embed_model=app_config.embed_model,
        dpi=app_config.dpi,
        num_workers=num_workers,
    )


def page_ranges(page_nums: list[int]) -> str:
    """Sorted page numbers with consecutive runs collapsed, e.g. [1, 2, 3, 7] -> 'pp. 1-3, 7'."""
    runs: list[list[int]] = []
    for n in sorted(page_nums):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    return ("p. " if len(page_nums) == 1 else "pp. ") + text


def print_plan(plan: Plan, params: SyncParams, *, detailed: bool = False) -> None:
    """Print what a sync would do.

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
            groups.setdefault((job.id.source_file, job.reason), []).append(job.id.page_num)
        for (source_file, reason), page_nums in sorted(groups.items()):
            typer.echo(f"    {source_file}  {page_ranges(page_nums)}  ({reason})")

    typer.echo(f"prompt {params.prompt.version}, {params.claude_model}, {params.embed_model}")
    line("extract", plan.extract)
    line("re-embed", plan.embed)
    typer.echo(f"  {'delete':<10} {len(plan.delete):>4}")
    for page_id in plan.delete:
        typer.echo(f"    {page_id}")
    typer.echo(f"  {'unchanged':<10} {plan.unchanged:>4}")


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

    conn = connect()

    with conn:
        plan = plan_sync(conn, target, params)
        print_plan(plan, params, detailed=dry_run)

        if dry_run or plan.is_empty():
            return
        if plan.extract and not os.environ.get("ANTHROPIC_API_KEY"):
            raise fail("ANTHROPIC_API_KEY is not set")
        if plan.delete and not yes:
            typer.confirm(f"Delete {len(plan.delete)} pages from the database?", abort=True)

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

        summary = apply_plan(
            conn,
            plan,
            params,
            extractor,
            embedder,
        )

    typer.echo(f"\n{summary.extracted} extracted, {summary.embedded} re-embedded, "
               f"{summary.deleted} deleted, {len(summary.failed)} failed")
    for failure in summary.failed:
        typer.echo(f"  {failure.job.id}: {failure.error}",
                   err=True)

    if summary.failed:
        raise typer.Exit(1)


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

    conn = connect()

    with conn:
        embedder = Embedder(APP_CONFIG.embed_model, APP_CONFIG.query_prefix)
        results = run_search(conn, embedder, query, k)

    if not results:
        typer.echo(f"No pages embedded with {APP_CONFIG.embed_model} yet. Run `sync` first.")
        return

    width = min(shutil.get_terminal_size().columns, 100)

    def indented(text: str) -> str:
        return textwrap.fill(text, width=width, initial_indent="   ", subsequent_indent="   ")

    for rank, r in enumerate(results, start=1):
        typer.echo(f"{rank}. {r.score:.3f}  {r.id}")
        typer.echo(indented(textwrap.shorten(r.summary, width=160)))
        typer.echo()
