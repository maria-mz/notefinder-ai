import asyncio
import logging
from dataclasses import dataclass

import db
from extract import Extraction, Extractor

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 4


@dataclass
class Job:
    source_file: str
    page_num: int
    content_hash: str
    png_bytes: bytes


@dataclass
class JobSuccess:
    job: Job
    extraction: Extraction
    embedding: list[float]


@dataclass
class JobFailure:
    job: Job
    error: str


JobResult = JobSuccess | JobFailure


def embedding_text(extraction: Extraction) -> str:
    """What gets embedded. Changing this means re-embedding every page."""
    return f"{extraction.summary} Concepts: {', '.join(extraction.concepts)}"


def describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


async def process(extractor: Extractor, embedder, prompt: str, job: Job) -> JobResult:
    """The slow part: read the page and embed it. Touches no database state."""
    try:
        extraction = await extractor.extract(job.png_bytes, prompt)
        vectors = embedder.embed_documents([embedding_text(extraction)])
    except Exception as e:
        return JobFailure(job, describe(e))
    return JobSuccess(job, extraction, vectors[0])


def log_result(result: JobResult) -> None:
    job = result.job
    if isinstance(result, JobFailure):
        logger.warning("failed %s p.%d: %s", job.source_file, job.page_num, result.error)
    else:
        e = result.extraction
        logger.info("done %s p.%d (%d in / %d out tokens)",
                    job.source_file, job.page_num, e.input_tokens, e.output_tokens)


async def run_worker(conn, extractor: Extractor, embedder, prompt: str, prompt_version: str,
                     job_queue: asyncio.Queue[Job | None], results: list[JobResult]) -> None:
    while True:
        job = await job_queue.get()
        if job is None:
            return
        result = await process(extractor, embedder, prompt, job)
        if isinstance(result, JobSuccess):
            db.save_page(
                conn, job.source_file, job.page_num, job.content_hash,
                transcript=result.extraction.transcript,
                concepts=result.extraction.concepts,
                summary=result.extraction.summary,
                prompt_version=prompt_version,
                embedding=result.embedding,
            )
        log_result(result)
        results.append(result)


async def run_pipeline(conn, extractor: Extractor, embedder, jobs: list[Job], *, prompt: str,
                       prompt_version: str, num_workers: int = DEFAULT_WORKERS) -> list[JobResult]:
    """Process every job. Returns one result per job, in completion order."""
    assert num_workers >= 1
    job_queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=num_workers)
    results: list[JobResult] = []

    async with asyncio.TaskGroup() as tg:
        for _ in range(num_workers):
            tg.create_task(
                run_worker(conn, extractor, embedder, prompt, prompt_version, job_queue, results)
            )

        for job in jobs:
            await job_queue.put(job)

        for _ in range(num_workers):
            await job_queue.put(None)
    # leaving the block waits for every worker to finish

    return results
