import pytest
from notefinder import db
from tests.helpers import axis_vector


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


def save_example_page(
    conn,
    *,
    page_id=PAGE_ID,
    content_hash=CONTENT_HASH,
    results=None,
):
    """Save a valid page with overridable identity and extraction results."""
    db.save_page(
        conn,
        page_id,
        content_hash,
        **(RESULTS if results is None else results),
    )


def assert_page_matches_results(page, *, page_id, content_hash, results):
    """Check persisted page fields, allowing for pgvector float precision."""
    assert page is not None
    assert page.id == page_id
    assert page.content_hash == content_hash
    assert page.transcript == results["transcript"]
    assert page.concepts == results["concepts"]
    assert page.summary == results["summary"]
    assert page.prompt_version == results["prompt_version"]
    assert page.claude_model == results["claude_model"]
    assert page.embedded_text == results["embedded_text"]
    assert page.embed_model == results["embed_model"]
    assert page.embedding == pytest.approx(results["embedding"])


def test_get_page_returns_none_when_page_does_not_exist(conn):
    assert db.get_page(conn, PAGE_ID) is None


def test_save_page_inserts_a_page_with_all_results(conn):
    save_example_page(conn)

    page = db.get_page(conn, PAGE_ID)

    assert_page_matches_results(
        page,
        page_id=PAGE_ID,
        content_hash=CONTENT_HASH,
        results=RESULTS,
    )


def test_save_page_overwrites_existing_results_for_the_same_page(conn):
    save_example_page(conn)
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
    updated_hash = "hash-2"

    save_example_page(
        conn,
        content_hash=updated_hash,
        results=updated_results,
    )

    page = db.get_page(conn, PAGE_ID)
    assert_page_matches_results(
        page,
        page_id=PAGE_ID,
        content_hash=updated_hash,
        results=updated_results,
    )


def test_get_page_states_returns_what_a_sync_compares(conn):
    save_example_page(conn, page_id=db.PageId("lecture-a.pdf", 1), content_hash="a1")
    save_example_page(conn, page_id=db.PageId("lecture-a.pdf", 2), content_hash="a2")

    states = sorted(db.get_page_states(conn), key=lambda s: s.id)

    assert states == [
        db.PageState(
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


def test_get_page_states_returns_empty_list_when_table_is_empty(conn):
    assert db.get_page_states(conn) == []


@pytest.mark.parametrize(
    ("field", "empty_value"),
    [
        ("content_hash", ""),
        ("transcript", ""),
        ("concepts", []),
        ("summary", ""),
        ("prompt_version", ""),
        ("claude_model", ""),
        ("embedded_text", ""),
        ("embed_model", ""),
        ("embedding", []),
    ],
)
def test_save_page_rejects_empty_required_values_without_inserting_a_page(
    conn, field, empty_value
):
    results = {**RESULTS, field: empty_value}

    with pytest.raises(AssertionError, match="Results must not be empty"):
        db.save_page(
            conn,
            PAGE_ID,
            results.pop("content_hash", CONTENT_HASH),
            **results,
        )

    assert db.get_page(conn, PAGE_ID) is None


def test_save_embedding_replaces_the_embedding_and_keeps_the_extraction(conn):
    save_example_page(conn)

    db.save_embedding(conn, PAGE_ID, embedded_text="New text.",
                      embed_model="embed-test-2", embedding=[0.5] * 1024)

    page = db.get_page(conn, PAGE_ID)
    assert_page_matches_results(
        page,
        page_id=PAGE_ID,
        content_hash=CONTENT_HASH,
        results={**RESULTS, "embedded_text": "New text.", "embed_model": "embed-test-2",
                 "embedding": [0.5] * 1024},
    )


@pytest.mark.parametrize(
    ("field", "empty_value"),
    [("embedded_text", ""), ("embed_model", ""), ("embedding", [])],
)
def test_save_embedding_rejects_empty_values_without_changing_the_page(
    conn, field, empty_value
):
    save_example_page(conn)
    new = {"embedded_text": "New text.", "embed_model": "embed-test-2",
           "embedding": [0.5] * 768, field: empty_value}

    with pytest.raises(AssertionError, match="Results must not be empty"):
        db.save_embedding(conn, PAGE_ID, **new)

    assert db.get_page(conn, PAGE_ID).embed_model == RESULTS["embed_model"]


def test_delete_pages_deletes_only_the_given_pages(conn):
    for page_id in [db.PageId("a.pdf", 1), db.PageId("a.pdf", 2), db.PageId("b.pdf", 1)]:
        save_example_page(conn, page_id=page_id)

    db.delete_pages(conn, [db.PageId("a.pdf", 2), db.PageId("b.pdf", 1)])

    assert [s.id for s in db.get_page_states(conn)] == [db.PageId("a.pdf", 1)]


def test_delete_pages_with_no_keys_does_nothing(conn):
    save_example_page(conn)

    db.delete_pages(conn, [])

    assert db.get_page(conn, PAGE_ID) is not None


def test_page_id_prints_as_file_and_page():
    assert str(db.PageId("os/lecture1.pdf", 3)) == "os/lecture1.pdf p.3"


def test_page_ids_sort_by_file_then_page_number():
    ids = [db.PageId("b.pdf", 1), db.PageId("a.pdf", 10), db.PageId("a.pdf", 2)]

    assert sorted(ids) == [db.PageId("a.pdf", 2), db.PageId("a.pdf", 10), db.PageId("b.pdf", 1)]


def test_page_ids_are_hashable_keys():
    assert {db.PageId("a.pdf", 1): "x"}[db.PageId("a.pdf", 1)] == "x"


def test_pgvector_conversion_round_trips_float_values():
    vector = [0.0, -1.25, 3.5]

    encoded = db.to_pgvector(vector)

    assert encoded == "[0.0,-1.25,3.5]"
    assert db.from_pgvector(encoded) == pytest.approx(vector)


def save_page_with_embedding(conn, source_file, embedding, summary="Summary.",
                             embed_model=RESULTS["embed_model"]):
    save_example_page(
        conn,
        page_id=db.PageId(source_file, PAGE_ID.page_num),
        results={**RESULTS, "summary": summary, "embedding": embedding,
                 "embed_model": embed_model},
    )


def search(conn, query_vector, limit):
    return db.search(conn, query_vector, limit=limit, embed_model=RESULTS["embed_model"])


def test_search_returns_pages_nearest_first(conn):
    save_page_with_embedding(conn, "far.pdf", axis_vector(0, 1))
    save_page_with_embedding(conn, "near.pdf", axis_vector(1, 0))
    save_page_with_embedding(conn, "middle.pdf", axis_vector(1, 1))

    results = search(conn, axis_vector(1, 0), limit=3)

    assert [r.id.source_file for r in results] == ["near.pdf", "middle.pdf", "far.pdf"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.7071, 0.0], abs=1e-4)


def test_search_returns_at_most_limit_pages(conn):
    for i in range(1, 4):
        save_page_with_embedding(conn, f"lecture-{i}.pdf", axis_vector(1, i))

    assert len(search(conn, axis_vector(1, 0), limit=2)) == 2


def test_search_returns_the_page_summary_and_concepts(conn):
    save_page_with_embedding(conn, PAGE_ID.source_file, axis_vector(1, 0), summary="Logic notes.")

    [result] = search(conn, axis_vector(1, 0), limit=5)

    assert result.id == PAGE_ID
    assert result.summary == "Logic notes."
    assert result.concepts == RESULTS["concepts"]


def test_search_returns_nothing_when_table_is_empty(conn):
    assert search(conn, axis_vector(1, 0), limit=5) == []


def test_search_skips_pages_embedded_with_another_model(conn):
    """Vectors from different models can't be compared, even when the dimensions differ."""
    save_page_with_embedding(conn, "current.pdf", axis_vector(0, 1))
    save_page_with_embedding(conn, "old.pdf", [1.0, 0.0, 0.0], embed_model="old-model")

    results = search(conn, axis_vector(1, 0), limit=5)

    assert [r.id.source_file for r in results] == ["current.pdf"]
