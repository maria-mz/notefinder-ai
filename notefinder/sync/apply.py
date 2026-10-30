"""Carrying out a sync plan: deleting, extracting, and re-embedding pages."""
import asyncio
import logging
from dataclasses import dataclass

from notefinder import db
from notefinder.embeddings import Embedder
from notefinder.extract import Extraction, Extractor
from notefinder.render import hash_image, render_page
from notefinder.sync.plan import EmbedJob, ExtractJob, Job, Plan, SyncParams, embedding_text

logger = logging.getLogger(__name__)


@dataclass
class Extracted:
    """The result of a successful :class:`ExtractJob`.

    :ivar job: The job that produced this result.
    :ivar content_hash: SHA-256 of the page image Claude read.
    :ivar extraction: What Claude extracted from the page.
    :ivar embedded_text: The exact text that was embedded.
    :ivar embedding: The vector for ``embedded_text``.
    """
    job: ExtractJob
    content_hash: str
    extraction: Extraction
    embedded_text: str
    embedding: list[float]


@dataclass
class Embedded:
    """The result of a successful :class:`EmbedJob`.

    :ivar job: The job that produced this result.
    :ivar embedded_text: The exact text that was embedded.
    :ivar embedding: The vector for ``embedded_text``.
    """
    job: EmbedJob
    embedded_text: str
    embedding: list[float]


@dataclass
class JobFailure:
    """A job that failed. Nothing was saved for it.

    :ivar job: The job that failed.
    :ivar error: The error.
    """
    job: Job
    error: str


#: The outcome of processing one job.
JobResult = Extracted | Embedded | JobFailure


def describe(error: Exception) -> str:
    """Format an error for reports and logs.

    :param error: The error to format.
    :returns: The error's type name and message, e.g. ``"RuntimeError: API down"``.
    """
    return f"{type(error).__name__}: {error}"


async def process(
    job: Job,
    extractor: Extractor | None,
    embedder: Embedder | None,
    params: SyncParams
) -> JobResult:
    """Extract and embed a page, or re-embed it. Touches no database state.

    :param job: The job to process.
    :param extractor: Reads the page. May be ``None`` if ``job`` is an :class:`EmbedJob`.
    :param embedder: Embeds the page's text.
    :param params: Supplies the prompt and the DPI to render at.
    :returns: :class:`Extracted` or :class:`Embedded` on success, or :class:`JobFailure`
        if any step raises. Never raises itself.
    """
    try:
        if isinstance(job, EmbedJob):
            text = embedding_text(job.summary, job.concepts)
            return Embedded(job, text, embedder.embed_documents([text])[0])

        png = render_page(job.pdf_path, job.id.page_num, params.dpi)

        extraction = await extractor.extract(png, params.prompt)
        text = embedding_text(extraction.summary, extraction.concepts)

        return Extracted(job, hash_image(png), extraction, text,
                         embedder.embed_documents([text])[0])
    except Exception as e:
        return JobFailure(job, describe(e))


def save(conn, result: Extracted | Embedded, params: SyncParams) -> None:
    """Save a successful result to the database.

    An :class:`Extracted` result replaces the whole page row. An :class:`Embedded`
    result replaces only the page's embedding.

    :param conn: The database connection.
    :param result: The result to save.
    :param params: Supplies the prompt version and model names to record.
    :raises psycopg.Error: If the database write fails.
    """
    job = result.job
    if isinstance(result, Embedded):
        db.save_embedding(
            conn, job.id,
            embedded_text=result.embedded_text,
            embed_model=params.embed_model,
            embedding=result.embedding,
        )
        return

    db.save_page(
        conn, job.id, result.content_hash,
        transcript=result.extraction.transcript,
        concepts=result.extraction.concepts,
        summary=result.extraction.summary,
        prompt_version=params.prompt.version,
        claude_model=params.claude_model,
        embedded_text=result.embedded_text,
        embed_model=params.embed_model,
        embedding=result.embedding,
    )


def log_result(result: JobResult, done: int, total: int) -> None:
    """Log one line for a result: a warning for a failure, info otherwise.

    :param result: The result to log.
    :param done: How many jobs have finished, this one included.
    :param total: How many jobs the sync has.
    """
    job = result.job
    progress = f"[{done:>{len(str(total))}}/{total}]"
    if isinstance(result, JobFailure):
        logger.warning("%s failed %s: %s", progress, job.id, result.error)
    elif isinstance(result, Embedded):
        logger.info("%s re-embedded %s", progress, job.id)
    else:
        e = result.extraction
        logger.info("%s extracted %s (%d in / %d out tokens)",
                    progress, job.id, e.input_tokens, e.output_tokens)


async def run_worker(
    conn,
    extractor: Extractor | None,
    embedder,
    params: SyncParams,
    job_queue: asyncio.Queue[Job | None],
    results: list[JobResult],
    total: int,
) -> None:
    """Process jobs from the queue until it yields ``None``.

    Saves each success, logs every result, and appends it to ``results``.

    :param conn: The database connection.
    :param extractor: Passed to :func:`process`.
    :param embedder: Passed to :func:`process`.
    :param params: Passed to :func:`process` and :func:`save`.
    :param job_queue: Jobs to process, then ``None`` to stop.
    :param results: Collects one result per job.
    :param total: How many jobs there are in all, across every worker. For progress logging.
    :raises psycopg.Error: If saving a result fails.
    """
    while True:
        job = await job_queue.get()
        if job is None:
            return

        result = await process(job, extractor, embedder, params)

        if not isinstance(result, JobFailure):
            save(conn, result, params)

        results.append(result)
        log_result(result, len(results), total)


async def run_jobs(
    conn,
    extractor: Extractor | None,
    embedder: Embedder | None,
    jobs: list[Job],
    *,
    params: SyncParams,
) -> list[JobResult]:
    """Process every job with ``params.num_workers`` concurrent workers.

    Each result is saved as soon as it's ready, so an interrupted run keeps
    the pages already finished.

    :param conn: The database connection.
    :param extractor: Reads pages. May be ``None`` if no job is an :class:`ExtractJob`.
    :param embedder: Embeds pages. May be ``None`` if ``jobs`` is empty.
    :param jobs: The jobs to process.
    :param params: The settings to process with and record. ``num_workers`` must be
        at least 1.
    :returns: One result per job, in completion order.
    :raises ExceptionGroup: If a worker fails, e.g. on a database error.
    """
    num_workers = params.num_workers
    assert num_workers >= 1
    job_queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=num_workers)
    results: list[JobResult] = []

    async with asyncio.TaskGroup() as tg:
        for _ in range(num_workers):
            tg.create_task(
                run_worker(conn, extractor, embedder, params, job_queue, results, len(jobs))
            )

        for job in jobs:
            await job_queue.put(job)

        for _ in range(num_workers):
            await job_queue.put(None)
    # leaving the block waits for every worker to finish

    return results


@dataclass
class SyncSummary:
    """What a sync did.

    :ivar extracted: Pages extracted and embedded.
    :ivar embedded: Pages re-embedded from their stored extraction.
    :ivar deleted: Pages deleted because they're gone from the notes.
    :ivar unchanged: Pages already up to date.
    :ivar failed: Jobs that failed. Nothing was saved for them.
    """
    extracted: int
    embedded: int
    deleted: int
    unchanged: int
    failed: list[JobFailure]


def apply_plan(
    conn,
    plan: Plan,
    params: SyncParams,
    extractor: Extractor | None,
    embedder: Embedder | None,
) -> SyncSummary:
    """Delete, extract, and re-embed pages as planned.

    :param conn: The database connection.
    :param plan: The plan to carry out.
    :param params: The settings to process with and record.
    :param extractor: Reads pages. May be ``None`` if the plan extracts nothing.
    :param embedder: Embeds pages. May be ``None`` if the plan has no jobs.
    :returns: A count of what was done, and the jobs that failed.
    :raises ExceptionGroup: If a worker fails, e.g. on a database error.
    """
    db.delete_pages(conn, plan.delete)
    for page_id in plan.delete:
        logger.info("deleted %s", page_id)

    results = asyncio.run(run_jobs(conn, extractor, embedder, plan.jobs, params=params))

    return SyncSummary(
        extracted=sum(isinstance(r, Extracted) for r in results),
        embedded=sum(isinstance(r, Embedded) for r in results),
        deleted=len(plan.delete),
        unchanged=plan.unchanged,
        failed=[r for r in results if isinstance(r, JobFailure)],
    )
