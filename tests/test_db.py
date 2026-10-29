import pytest
import db


SOURCE_FILE = "test_notes.pdf"
HASH_1 = "hash-1"
HASH_2 = "hash-2"
RESULTS = {
    "transcript": "∀x P(x)",
    "concepts": ["Predicate logic", "Quantifiers"],
    "summary": "Predicate logic notes.",
    "prompt_version": "v1",
    "embedding": [i / 1024 for i in range(768)],
}


def insert_pending_page(conn, source_file=SOURCE_FILE, page_num=1,
                        content_hash=HASH_1):
    """Insert a page and verify that it starts in the pending state."""
    result = db.upsert_page(conn, source_file, page_num, content_hash)
    assert result is db.UpsertResult.NEW
    return db.get_page(conn, source_file, page_num)


def insert_completed_page(conn, source_file=SOURCE_FILE, page_num=1,
                          content_hash=HASH_1):
    """Create a page with results."""
    insert_pending_page(conn, source_file, page_num, content_hash)
    claimed = db.claim_page(conn, source_file, page_num)
    assert claimed is not None
    return db.mark_done(conn, source_file, page_num, **RESULTS)


def test_upsert_page_inserts_a_pending_page(conn):
    result = db.upsert_page(conn, SOURCE_FILE, 1, HASH_1)

    assert result is db.UpsertResult.NEW
    assert db.get_page(conn, SOURCE_FILE, 1) == db.Page(
        source_file=SOURCE_FILE,
        page_num=1,
        content_hash=HASH_1,
    )


def test_get_page_returns_none_when_the_page_does_not_exist(conn):
    assert db.get_page(conn, SOURCE_FILE, 99) is None


def test_upsert_page_preserves_results_when_the_hash_is_unchanged(conn):
    completed = insert_completed_page(conn)

    result = db.upsert_page(conn, SOURCE_FILE, 1, HASH_1)

    assert result is db.UpsertResult.UNCHANGED
    assert db.get_page(conn, SOURCE_FILE, 1) == completed


def test_upsert_page_resets_processing_results_when_the_hash_changes(conn):
    insert_completed_page(conn)
    # Seed stale metadata as well as completed results so the reset covers all
    # fields that upsert_page promises to clear.
    conn.execute(
        "UPDATE pages SET attempts = 4, last_error = %s "
        "WHERE source_file = %s AND page_num = %s",
        ("stale error", SOURCE_FILE, 1),
    )

    result = db.upsert_page(conn, SOURCE_FILE, 1, HASH_2)

    assert result is db.UpsertResult.CHANGED
    assert db.get_page(conn, SOURCE_FILE, 1) == db.Page(
        source_file=SOURCE_FILE,
        page_num=1,
        content_hash=HASH_2,
    )


def test_claim_page_claims_pending_page_only_once(conn):
    insert_pending_page(conn)

    claimed = db.claim_page(conn, SOURCE_FILE, 1)

    assert claimed is not None
    assert claimed.status == "processing"
    assert claimed.attempts == 1
    assert claimed.last_error is None

    # A processing page is no longer eligible for another claim.
    assert db.claim_page(conn, SOURCE_FILE, 1) is None
    assert db.get_page(conn, SOURCE_FILE, 1).attempts == 1


def test_claim_page_returns_none_for_an_unknown_page(conn):
    assert db.claim_page(conn, SOURCE_FILE, 99) is None


def test_mark_done_persists_all_extraction_results(conn):
    insert_pending_page(conn)
    assert db.claim_page(conn, SOURCE_FILE, 1) is not None

    completed = db.mark_done(conn, SOURCE_FILE, 1, **RESULTS)
    persisted = db.get_page(conn, SOURCE_FILE, 1)

    assert completed == persisted
    assert persisted.status == "done"
    assert persisted.attempts == 1
    assert persisted.last_error is None
    assert persisted.transcript == RESULTS["transcript"]
    assert persisted.concepts == RESULTS["concepts"]
    assert persisted.summary == RESULTS["summary"]
    assert persisted.prompt_version == RESULTS["prompt_version"]
    assert persisted.embedding == pytest.approx(RESULTS["embedding"])


def test_mark_done_rejects_an_empty_result_field(conn):
    insert_pending_page(conn)
    assert db.claim_page(conn, SOURCE_FILE, 1) is not None
    results = {**RESULTS, "transcript": ""}

    with pytest.raises(AssertionError, match="Results must not be empty"):
        db.mark_done(conn, SOURCE_FILE, 1, **results)

    # Rejected results must not transition the page to done.
    assert db.get_page(conn, SOURCE_FILE, 1).status == "processing"


def test_mark_done_rejects_a_page_that_is_not_processing(conn):
    insert_pending_page(conn)

    with pytest.raises(AssertionError, match="not in 'processing'"):
        db.mark_done(conn, SOURCE_FILE, 1, **RESULTS)

    assert db.get_page(conn, SOURCE_FILE, 1).status == "pending"


def test_mark_failed_records_the_error_for_a_processing_page(conn):
    insert_pending_page(conn)
    assert db.claim_page(conn, SOURCE_FILE, 1) is not None

    failed = db.mark_failed(conn, SOURCE_FILE, 1, "model request timed out")

    assert failed.status == "failed"
    assert failed.attempts == 1
    assert failed.last_error == "model request timed out"
    assert failed.transcript is None
    assert db.get_page(conn, SOURCE_FILE, 1) == failed


def test_mark_failed_rejects_a_page_that_is_not_processing(conn):
    insert_pending_page(conn)

    with pytest.raises(AssertionError, match="not in 'processing'"):
        db.mark_failed(conn, SOURCE_FILE, 1, "unexpected failure")

    assert db.get_page(conn, SOURCE_FILE, 1).status == "pending"


def test_reset_stuck_resets_only_processing_pages(conn):
    for page_num in range(1, 5):
        insert_pending_page(conn, page_num=page_num)

    # Pages 1 and 2 are left processing, page 3 fails, and page 4 stays pending.
    assert db.claim_page(conn, SOURCE_FILE, 1) is not None
    assert db.claim_page(conn, SOURCE_FILE, 2) is not None
    assert db.claim_page(conn, SOURCE_FILE, 3) is not None
    db.mark_failed(conn, SOURCE_FILE, 3, "temporary error")

    reset_count = db.reset_stuck(conn)

    assert reset_count == 2
    assert db.get_page(conn, SOURCE_FILE, 1).status == "pending"
    assert db.get_page(conn, SOURCE_FILE, 1).attempts == 1
    assert db.get_page(conn, SOURCE_FILE, 2).status == "pending"
    assert db.get_page(conn, SOURCE_FILE, 3).status == "failed"
    assert db.get_page(conn, SOURCE_FILE, 3).last_error == "temporary error"
    assert db.get_page(conn, SOURCE_FILE, 4).status == "pending"
    assert db.get_page(conn, SOURCE_FILE, 4).attempts == 0


def test_pgvector_conversion_round_trips_float_values():
    vector = [0.0, -1.25, 3.5]

    encoded = db.to_pgvector(vector)

    assert encoded == "[0.0,-1.25,3.5]"
    assert db.from_pgvector(encoded) == pytest.approx(vector)
