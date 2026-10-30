import logging
import os
from pathlib import Path
from typing import Annotated

import psycopg
import typer

import db
from extract import Extractor, make_client
from ingest import load_prompt, validate_target
from ingest import ingest as run_ingest
from pipeline import DEFAULT_WORKERS

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

NotesDir = Annotated[Path, typer.Option(
    envvar="NOTEFINDER_NOTES_DIR",
    help="Root folder of the notes. Keep it the same between runs: "
         "pages are stored by their path relative to it.",
)]

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Search handwritten notes by topic."""
    # a callback keeps `ingest` a named subcommand while it is the only one
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # logs every API request


def fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


@app.command()
def ingest(
    notes_dir: NotesDir,
    target: Annotated[Path | None, typer.Argument(
        help="A PDF or subfolder of the notes to ingest. Default: all of them.",
    )] = None,
    workers: Annotated[int, typer.Option(min=1)] = DEFAULT_WORKERS,
) -> None:
    """Extract and embed every new or changed page."""
    # cheap checks first, before loading the model or rendering any pages
    try:
        validate_target(notes_dir, target)
    except (FileNotFoundError, ValueError) as e:
        raise fail(str(e), code=2)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise fail("ANTHROPIC_API_KEY is not set")

    prompt, prompt_version = load_prompt(PROMPT_PATH)

    try:
        conn = db.connect()
    except psycopg.OperationalError as e:
        raise fail(f"could not connect to the database (is Docker running?)\n{e}")

    # sentence-transformers takes seconds to import; keep it out of --help
    from embeddings import Embedder

    with conn:
        summary = run_ingest(
            conn,
            notes_dir,
            Extractor(make_client()),
            Embedder(),
            prompt=prompt,
            prompt_version=prompt_version,
            num_workers=workers,
            target=target,
        )

    typer.echo(f"\n{summary.saved} saved, {summary.unchanged} unchanged, "
               f"{len(summary.failed)} failed (prompt {prompt_version})")
    for failure in summary.failed:
        typer.echo(f"  {failure.job.source_file} p.{failure.job.page_num}: {failure.error}",
                   err=True)

    if summary.failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
