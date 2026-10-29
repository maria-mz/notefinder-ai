import asyncio

import psycopg
import pytest

import db
from extract import Extraction
from pipeline import JobFailure, Job, JobSuccess, embedding_text, process, run_pipeline

SOURCE_FILE = "lecture.pdf"
PROMPT = "Transcribe this page."
VERSION = "v1"


def mock_extraction(png_bytes: bytes) -> Extraction:
    name = png_bytes.decode()
    return Extraction(
        transcript=f"transcript of {name}",
        concepts=[f"concept of {name}", "Another concept"],
        summary=f"summary of {name}",
        input_tokens=100,
        output_tokens=50,
    )


def mock_vector(text: str) -> list[float]:
    """A deterministic 768-number 'embedding'."""
    return [(sum(map(ord, text)) % 997) / 1000] * 768


class MockExtractor:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def extract(self, png_bytes: bytes, prompt: str) -> Extraction:
        if self.fail:
            raise RuntimeError("API down")
        return mock_extraction(png_bytes)


class MockEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed")
        return [mock_vector(text) for text in texts]


def make_job(page_num: int) -> Job:
    return Job(SOURCE_FILE, page_num, f"hash-{page_num}", f"png-{page_num}".encode())


def run(conn, extractor, embedder, jobs, num_workers=2):
    """Run the pipeline, but fail the test instead of hanging if it deadlocks."""
    return asyncio.run(asyncio.wait_for(
        run_pipeline(conn, extractor, embedder, jobs, prompt=PROMPT,
                     prompt_version=VERSION, num_workers=num_workers),
        timeout=10,
    ))


def run_process(extractor, embedder, job):
    return asyncio.run(process(extractor, embedder, PROMPT, job))


def saved_page_nums(conn) -> list[int]:
    return [n for (n,) in conn.execute("SELECT page_num FROM pages ORDER BY page_num")]


def test_embedding_text_format():
    extraction = Extraction(transcript="t", concepts=["A", "B"], summary="A summary.",
                            input_tokens=0, output_tokens=0)
    assert embedding_text(extraction) == "A summary. Concepts: A, B"


def test_process_succeeds_with_the_extraction_and_embedding():
    job = make_job(1)
    expected_extraction = mock_extraction(job.png_bytes)

    result = run_process(MockExtractor(), MockEmbedder(), job)
    e = result.extraction

    assert isinstance(result, JobSuccess)
    assert result.job == job
    assert e.transcript == expected_extraction.transcript
    assert e.concepts == expected_extraction.concepts
    assert e.summary == expected_extraction.summary
    assert result.embedding == pytest.approx(
        mock_vector(embedding_text(expected_extraction))
    )


def test_process_fails_when_extraction_fails():
    result = run_process(MockExtractor(fail=True), MockEmbedder(), make_job(1))
    assert result == JobFailure(make_job(1), "RuntimeError: API down")


def test_process_fails_when_embedding_fails():
    result = run_process(MockExtractor(), MockEmbedder(fail=True), make_job(1))
    assert result == JobFailure(make_job(1), "RuntimeError: embedding failed")


@pytest.mark.parametrize("num_jobs, num_workers", [
    pytest.param(0, 2, id="no-jobs"),
    pytest.param(1, 1, id="one-job-one-worker"),
    pytest.param(2, 5, id="more-workers-than-jobs"),
    pytest.param(10, 3, id="more-jobs-than-workers"),
])
def test_every_job_is_saved_with_its_own_results(conn, num_jobs, num_workers):
    jobs = [make_job(n) for n in range(1, num_jobs + 1)]
    run(conn, MockExtractor(), MockEmbedder(), jobs, num_workers)

    assert saved_page_nums(conn) == list(range(1, num_jobs + 1))
    for job in jobs:
        page = db.get_page(conn, job.source_file, job.page_num)

        assert page.content_hash == job.content_hash
        assert page.prompt_version == VERSION

        expected_extraction = mock_extraction(job.png_bytes)
        assert page.transcript == expected_extraction.transcript
        assert page.concepts == expected_extraction.concepts
        assert page.summary == expected_extraction.summary
        assert page.embedding == pytest.approx(
            mock_vector(embedding_text(expected_extraction))
        )


def test_a_failed_rerun_keeps_the_pages_old_results(conn):
    db.save_page(conn, SOURCE_FILE, 1, "old-hash", transcript="old transcript",
                 concepts=["Old"], summary="old summary", prompt_version="v0",
                 embedding=[0.5] * 768)

    run(conn, MockExtractor(fail=True), MockEmbedder(), [make_job(1)])

    page = db.get_page(conn, SOURCE_FILE, 1)
    assert page.content_hash == "old-hash"
    assert page.transcript == "old transcript"
    assert page.concepts == ["Old"]
    assert page.summary == "old summary"
    assert page.prompt_version == "v0"
    assert page.embedding == pytest.approx([0.5] * 768)


def test_database_error_propagates_from_worker(conn):
    conn.close()
    with pytest.raises(ExceptionGroup) as excinfo:
        run(conn, MockExtractor(), MockEmbedder(), [make_job(n) for n in range(1, 6)])
    assert excinfo.value.subgroup(psycopg.Error) is not None


def test_zero_workers_is_rejected(conn):
    with pytest.raises(AssertionError):
        run(conn, MockExtractor(), MockEmbedder(), [make_job(1)], num_workers=0)
