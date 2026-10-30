import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import db
from extract import Extractor
from render import hash_image, page_count, relative_source, render_page
from pipeline import Job, JobFailure, run_pipeline, DEFAULT_WORKERS


@dataclass
class IngestSummary:
    unchanged: int
    saved: int
    failed: list[JobFailure]


def validate_target(
    root: Path,
    target: Path | None = None
) -> tuple[Path, Path | None]:
    root = root.resolve()
    target = root if target is None else target.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"notes folder not found: {root}")
    if not target.exists():
        raise FileNotFoundError(f"target not found: {target}")
    if not target.is_relative_to(root):
        raise ValueError(f"{target} is outside {root}")
    if target.is_file() and target.suffix != ".pdf":
        raise ValueError(f"{target} is not a PDF")

    return root, target


def find_pdfs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.pdf"))


def collect_jobs(
    root: Path,
    pdfs: list[Path],
    job_filter: Callable[[Job], bool],
) -> list[Job]:
    jobs: list[Job] = []

    for pdf in pdfs:
        source_file = relative_source(pdf, root)

        for page in range(1, page_count(pdf) + 1):
            png = render_page(pdf, page)
            job = Job(source_file, page, hash_image(png), png)
            if job_filter(job):
                jobs.append(job)

    return jobs


def ingest(
    conn,
    root: Path,
    extractor: Extractor,
    embedder,
    prompt: str,
    prompt_version: str,
    num_workers: int = DEFAULT_WORKERS,
    target: Path | None = None
) -> IngestSummary:
    """Extract and embed every new or changed page under target (default: all of root)."""
    root, target = validate_target(root, target)

    known = db.get_hashes(conn)
    pdfs = find_pdfs(target)

    def is_page_changed(job: Job) -> bool:
        return job.content_hash != known.get(
            (job.source_file, job.page_num)
        )

    jobs = collect_jobs(root, pdfs, job_filter=is_page_changed)

    total_pages = sum(page_count(pdf) for pdf in pdfs)

    results = asyncio.run(run_pipeline(
        conn,
        extractor,
        embedder,
        jobs,
        prompt=prompt,
        prompt_version=prompt_version,
        num_workers=num_workers
    ))

    failed = [r for r in results if isinstance(r, JobFailure)]
    return IngestSummary(
        unchanged=total_pages - len(jobs),
        saved=len(results) - len(failed),
        failed=failed,
    )
