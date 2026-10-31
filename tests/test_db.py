import pytest
from psycopg.errors import CheckViolation, DataException

from notefinder import db
from tests.helpers import PARAMS, axis_vector, save_test_page


PAGE_ID = db.PageId("lecture.pdf", 1)
CONTENT_HASH = "hash-1"
RESULTS = {
    "transcript": "∀x P(x)",
    "concepts": ["Predicate logic", "Quantifiers"],
    "summary": "Predicate logic notes.",
    "prompt_version": "p1-test",
    "claude_model": "claude-test",
    "embedded_text": "Predicate logic notes. Concepts: Predicate logic, Quantifiers",
    "embed_model": "embed-test",
    "embedding": [i / 1024 for i in range(768)],
}


def expected_page(results, content_hash=CONTENT_HASH) -> db.Page:
    """The page saved with these results. pgvector stores 32-bit floats, hence approx."""
    return db.Page(PAGE_ID, content_hash,
                   **{**results, "embedding": pytest.approx(results["embedding"])})


def test_get_page_returns_none_when_page_does_not_exist(conn):
    assert db.get_page(conn, PAGE_ID) is None


def test_save_page_inserts_a_page_with_all_results(conn):
    db.save_page(conn, PAGE_ID, CONTENT_HASH, **RESULTS)

    assert db.get_page(conn, PAGE_ID) == expected_page(RESULTS)


def test_save_page_overwrites_existing_results_for_the_same_page(conn):
    db.save_page(conn, PAGE_ID, CONTENT_HASH, **RESULTS)
    updated_results = {
        "transcript": "Updated transcript.",
        "concepts": ["Updated concept"],
        "summary": "Updated summary.",
        "prompt_version": "p2-test",
        "claude_model": "claude-test-2",
        "embedded_text": "Updated summary. Concepts: Updated concept",
        "embed_model": "embed-test-2",
        "embedding": [0.25] * 768,
    }

    db.save_page(conn, PAGE_ID, "hash-2", **updated_results)

    assert db.get_page(conn, PAGE_ID) == expected_page(updated_results, "hash-2")


def test_get_stored_pages_returns_what_a_sync_compares(conn):
    for n in (1, 2):
        db.save_page(conn, db.PageId("lecture-a.pdf", n), f"a{n}", **RESULTS)

    pages = sorted(db.get_stored_pages(conn), key=lambda s: s.id)

    assert pages == [
        db.StoredPage(
            id=db.PageId("lecture-a.pdf", n),
            content_hash=f"a{n}",
            prompt_version=RESULTS["prompt_version"],
            claude_model=RESULTS["claude_model"],
            embedded_text=RESULTS["embedded_text"],
            embed_model=RESULTS["embed_model"],
            summary=RESULTS["summary"],
            concepts=RESULTS["concepts"],
        )
        for n in (1, 2)
    ]


def test_get_stored_pages_returns_empty_list_when_table_is_empty(conn):
    assert db.get_stored_pages(conn) == []


@pytest.mark.parametrize(
    ("field", "empty_value", "error"),
    [
        ("content_hash", "", CheckViolation),
        ("transcript", "", CheckViolation),
        ("concepts", [], CheckViolation),
        ("summary", "", CheckViolation),
        ("prompt_version", "", CheckViolation),
        ("claude_model", "", CheckViolation),
        ("embedded_text", "", CheckViolation),
        ("embed_model", "", CheckViolation),
        ("embedding", [], DataException),  # pgvector rejects a vector with no dimensions
    ],
)
def test_save_page_rejects_empty_required_values_without_inserting_a_page(
    conn, field, empty_value, error
):
    fields = {"content_hash": CONTENT_HASH, **RESULTS, field: empty_value}

    with pytest.raises(error):
        db.save_page(conn, PAGE_ID, fields.pop("content_hash"), **fields)

    assert db.get_page(conn, PAGE_ID) is None


def test_save_embedding_replaces_the_embedding_and_keeps_the_extraction(conn):
    db.save_page(conn, PAGE_ID, CONTENT_HASH, **RESULTS)
    new = {"embedded_text": "New text.", "embed_model": "embed-test-2",
           "embedding": [0.5] * 1024}

    db.save_embedding(conn, PAGE_ID, **new)

    assert db.get_page(conn, PAGE_ID) == expected_page({**RESULTS, **new})


@pytest.mark.parametrize(
    ("field", "empty_value", "error"),
    [
        ("embedded_text", "", CheckViolation),
        ("embed_model", "", CheckViolation),
        ("embedding", [], DataException),
    ],
)
def test_save_embedding_rejects_empty_values_without_changing_the_page(
    conn, field, empty_value, error
):
    db.save_page(conn, PAGE_ID, CONTENT_HASH, **RESULTS)
    new = {"embedded_text": "New text.", "embed_model": "embed-test-2",
           "embedding": [0.5] * 768, field: empty_value}

    with pytest.raises(error):
        db.save_embedding(conn, PAGE_ID, **new)

    assert db.get_page(conn, PAGE_ID) == expected_page(RESULTS)


def test_delete_pages_deletes_only_the_given_pages(conn):
    for page_id in [db.PageId("a.pdf", 1), db.PageId("a.pdf", 2), db.PageId("b.pdf", 1)]:
        save_test_page(conn, page_id)

    db.delete_pages(conn, [db.PageId("a.pdf", 2), db.PageId("b.pdf", 1)])

    assert [s.id for s in db.get_stored_pages(conn)] == [db.PageId("a.pdf", 1)]


def test_delete_pages_with_no_keys_does_nothing(conn):
    save_test_page(conn, PAGE_ID)

    db.delete_pages(conn, [])

    assert db.get_page(conn, PAGE_ID) is not None


def test_page_id_prints_as_file_and_page():
    assert str(db.PageId("os/lecture1.pdf", 3)) == "os/lecture1.pdf p.3"


def test_page_ids_sort_by_file_then_page_number():
    ids = [db.PageId("b.pdf", 1), db.PageId("a.pdf", 10), db.PageId("a.pdf", 2)]

    assert sorted(ids) == [
        db.PageId("a.pdf", 2), db.PageId("a.pdf", 10), db.PageId("b.pdf", 1),
    ]


def test_pgvector_conversion_round_trips_float_values():
    vector = [0.0, -1.25, 3.5]

    encoded = db.to_pgvector(vector)

    assert encoded == "[0.0,-1.25,3.5]"
    assert db.from_pgvector(encoded) == pytest.approx(vector)


def search(conn, query_vector, limit):
    return db.search(conn, query_vector, limit=limit, embed_model=PARAMS.embed_model)


def test_search_returns_pages_nearest_first(conn):
    save_test_page(conn, db.PageId("far.pdf", 1), embedding=axis_vector(0, 1))
    save_test_page(conn, db.PageId("near.pdf", 1), embedding=axis_vector(1, 0))
    save_test_page(conn, db.PageId("middle.pdf", 1), embedding=axis_vector(1, 1))

    results = search(conn, axis_vector(1, 0), limit=3)

    assert [r.id.source_file for r in results] == ["near.pdf", "middle.pdf", "far.pdf"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.7071, 0.0], abs=1e-4)


def test_search_returns_at_most_limit_pages(conn):
    for i in range(1, 4):
        save_test_page(conn, db.PageId(f"lecture-{i}.pdf", 1))

    assert len(search(conn, axis_vector(1, 0), limit=2)) == 2


def test_search_returns_the_page_summary_and_concepts(conn):
    save_test_page(conn, PAGE_ID, summary="Logic notes.", concepts=["Logic"])

    [result] = search(conn, axis_vector(1, 0), limit=5)

    assert result.id == PAGE_ID
    assert result.summary == "Logic notes."
    assert result.concepts == ["Logic"]


def test_search_returns_nothing_when_table_is_empty(conn):
    assert search(conn, axis_vector(1, 0), limit=5) == []


def test_search_breaks_ties_by_page(conn):
    for page_id in [db.PageId("b.pdf", 1), db.PageId("a.pdf", 2), db.PageId("a.pdf", 1)]:
        save_test_page(conn, page_id)

    results = search(conn, axis_vector(1, 0), limit=3)

    assert [r.id for r in results] == [
        db.PageId("a.pdf", 1), db.PageId("a.pdf", 2), db.PageId("b.pdf", 1),
    ]


def test_search_skips_pages_embedded_with_another_model(conn):
    """Vectors from different models can't be compared, even when their sizes differ."""
    save_test_page(conn, db.PageId("current.pdf", 1), embedding=axis_vector(0, 1))
    save_test_page(conn, db.PageId("old.pdf", 1), embedding=[1.0, 0.0, 0.0],
                   embed_model="old-model")

    results = search(conn, axis_vector(1, 0), limit=5)

    assert [r.id.source_file for r in results] == ["current.pdf"]
