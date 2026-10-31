import asyncio
import logging
from dataclasses import replace

import psycopg
import pytest

from notefinder import db
from notefinder.db import PageId
from notefinder.extract import Extraction
from notefinder.sync.apply import Embedded, Extracted, JobFailure, process, run_jobs
from notefinder.sync.plan import (EmbedJob, EmbedReason, ExtractJob, ExtractReason,
                                  embedding_text)
from notefinder.render import hash_image, render_page
from tests.helpers import (PARAMS, MockEmbedder, MockExtractor, axis_vector, make_pdf,
                           mock_vector, save_test_page)

SOURCE_FILE = "lecture.pdf"
NUM_PAGES = 10


@pytest.fixture
def pdf(tmp_path):
    texts = [f"page {n}" for n in range(1, NUM_PAGES + 1)]
    return make_pdf(tmp_path / SOURCE_FILE, texts)


def mock_extraction(png_bytes: bytes) -> Extraction:
    """Different results for each page image."""
    name = hash_image(png_bytes)[:8]
    return Extraction(
        transcript=f"transcript of {name}",
        concepts=[f"concept of {name}", "Another concept"],
        summary=f"summary of {name}",
        input_tokens=100,
        output_tokens=50,
    )


def expected_extraction(pdf, page_num: int) -> Extraction:
    return mock_extraction(render_page(pdf, page_num, PARAMS.dpi))


def page_id(page_num: int) -> PageId:
    return PageId(SOURCE_FILE, page_num)


def extract_job(pdf, page_num: int) -> ExtractJob:
    return ExtractJob(page_id(page_num), pdf, ExtractReason.NEW)


def embed_job(page_num: int, summary: str = "New summary.") -> EmbedJob:
    return EmbedJob(page_id(page_num), summary, ["New concept"],
                    EmbedReason.EMBED_MODEL_CHANGED)


def run(conn, extractor, embedder, jobs, num_workers=2):
    """Run the jobs, but fail the test instead of hanging if it deadlocks."""
    return asyncio.run(asyncio.wait_for(
        run_jobs(conn, extractor, embedder, jobs,
                 params=replace(PARAMS, num_workers=num_workers)),
        timeout=10,
    ))


def run_process(extractor, embedder, job):
    return asyncio.run(process(job, extractor, embedder, PARAMS))


def saved_page_nums(conn) -> list[int]:
    return [n for (n,) in conn.execute("SELECT page_num FROM pages ORDER BY page_num")]


def test_process_extracts_and_embeds_the_rendered_page(pdf):
    job = extract_job(pdf, 2)
    extraction = expected_extraction(pdf, 2)
    text = embedding_text(extraction.summary, extraction.concepts)

    result = run_process(MockExtractor(mock_extraction), MockEmbedder(), job)

    assert result == Extracted(
        job,
        content_hash=hash_image(render_page(pdf, 2, PARAMS.dpi)),
        extraction=extraction,
        embedded_text=text,
        embedding=mock_vector(text),
    )


def test_process_re_embeds_without_an_extractor():
    job = embed_job(1)
    text = embedding_text("New summary.", ["New concept"])

    result = run_process(None, MockEmbedder(), job)

    assert result == Embedded(job, embedded_text=text, embedding=mock_vector(text))


def test_process_fails_when_extraction_fails(pdf):
    result = run_process(MockExtractor(fail=True), MockEmbedder(), extract_job(pdf, 1))
    assert result == JobFailure(extract_job(pdf, 1), "RuntimeError: API down")


@pytest.mark.parametrize("make_job", [
    pytest.param(lambda pdf: extract_job(pdf, 1), id="extract"),
    pytest.param(lambda pdf: embed_job(1), id="embed"),
])
def test_process_fails_when_embedding_fails(pdf, make_job):
    job = make_job(pdf)
    result = run_process(MockExtractor(), MockEmbedder(fail=True), job)
    assert result == JobFailure(job, "RuntimeError: embedding failed")


def test_process_fails_when_the_page_is_gone(pdf):
    """The PDF can lose pages between planning and processing."""
    result = run_process(MockExtractor(), MockEmbedder(), extract_job(pdf, NUM_PAGES + 1))
    assert isinstance(result, JobFailure)


@pytest.mark.parametrize("num_jobs, num_workers", [
    pytest.param(0, 2, id="no-jobs"),
    pytest.param(1, 1, id="one-job-one-worker"),
    pytest.param(2, 5, id="more-workers-than-jobs"),
    pytest.param(10, 3, id="more-jobs-than-workers"),
])
def test_every_extract_job_is_saved_with_its_own_results(conn, pdf, num_jobs,
                                                          num_workers):
    jobs = [extract_job(pdf, n) for n in range(1, num_jobs + 1)]
    run(conn, MockExtractor(mock_extraction), MockEmbedder(), jobs, num_workers)

    assert saved_page_nums(conn) == list(range(1, num_jobs + 1))
    for job in jobs:
        extraction = expected_extraction(pdf, job.id.page_num)
        text = embedding_text(extraction.summary, extraction.concepts)

        assert db.get_page(conn, job.id) == db.Page(
            id=job.id,
            content_hash=hash_image(render_page(pdf, job.id.page_num, PARAMS.dpi)),
            transcript=extraction.transcript,
            concepts=extraction.concepts,
            summary=extraction.summary,
            prompt_version=PARAMS.prompt.version,
            claude_model=PARAMS.claude_model,
            embedded_text=text,
            embed_model=PARAMS.embed_model,
            embedding=pytest.approx(mock_vector(text)),
        )


def test_an_embed_job_replaces_only_the_embedding(conn):
    save_test_page(conn, page_id(1), embed_model="old-model", embedding=[0.5] * 1024)
    before = db.get_page(conn, page_id(1))

    run(conn, None, MockEmbedder(), [embed_job(1, summary=before.summary)])

    text = embedding_text(before.summary, ["New concept"])
    assert db.get_page(conn, page_id(1)) == replace(
        before, embedded_text=text, embed_model=PARAMS.embed_model,
        embedding=pytest.approx(mock_vector(text)),
    )


def test_a_failed_rerun_keeps_the_pages_old_results(conn, pdf):
    save_test_page(conn, page_id(1), content_hash="old-hash", prompt_version="v0",
                   embedding=axis_vector(0, 1))
    before = db.get_page(conn, page_id(1))

    run(conn, MockExtractor(fail=True), MockEmbedder(), [extract_job(pdf, 1)])

    assert db.get_page(conn, page_id(1)) == before


def test_database_error_propagates_from_worker(conn, pdf):
    conn.close()
    with pytest.raises(ExceptionGroup) as excinfo:
        jobs = [extract_job(pdf, n) for n in range(1, 6)]
        run(conn, MockExtractor(), MockEmbedder(), jobs)
    assert excinfo.value.subgroup(psycopg.Error) is not None


def test_returns_one_result_per_job(conn, pdf):
    save_test_page(conn, page_id(3), embed_model="old-model")
    jobs = [extract_job(pdf, 1), extract_job(pdf, 2), embed_job(3)]

    results = run(conn, MockExtractor(), MockEmbedder(), jobs)

    assert sorted((r.job.id.page_num, type(r)) for r in results) == [
        (1, Extracted), (2, Extracted), (3, Embedded)]


def test_each_result_is_logged_with_its_progress(conn, pdf, caplog):
    jobs = [extract_job(pdf, n) for n in range(1, 11)]

    with caplog.at_level(logging.INFO, logger="notefinder.sync.apply"):
        run(conn, MockExtractor(mock_extraction), MockEmbedder(), jobs)

    lines = [record.getMessage() for record in caplog.records]
    assert sorted(line.split(" extracted")[0] for line in lines) == sorted(
        f"[{n:>2}/10]" for n in range(1, 11)
    )
