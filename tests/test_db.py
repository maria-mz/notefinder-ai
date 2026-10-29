import pytest
import db


SOURCE_FILE = "lecture.pdf"
PAGE_NUM = 1
CONTENT_HASH = "hash-1"
RESULTS = {
    "transcript": "∀x P(x)",
    "concepts": ["Predicate logic", "Quantifiers"],
    "summary": "Predicate logic notes.",
    "prompt_version": "p1-test",
    "embedding": [i / 1024 for i in range(768)],
}


def save_example_page(
    conn,
    *,
    source_file=SOURCE_FILE,
    page_num=PAGE_NUM,
    content_hash=CONTENT_HASH,
    results=None,
):
    """Save a valid page with overridable identity and extraction results."""
    db.save_page(
        conn,
        source_file,
        page_num,
        content_hash,
        **(RESULTS if results is None else results),
    )


def assert_page_matches_results(page, *, source_file, page_num, content_hash, results):
    """Check persisted page fields, allowing for pgvector float precision."""
    assert page is not None
    assert page.source_file == source_file
    assert page.page_num == page_num
    assert page.content_hash == content_hash
    assert page.transcript == results["transcript"]
    assert page.concepts == results["concepts"]
    assert page.summary == results["summary"]
    assert page.prompt_version == results["prompt_version"]
    assert page.embedding == pytest.approx(results["embedding"])


def test_get_page_returns_none_when_page_does_not_exist(conn):
    assert db.get_page(conn, SOURCE_FILE, PAGE_NUM) is None


def test_save_page_inserts_a_page_with_all_results(conn):
    save_example_page(conn)

    page = db.get_page(conn, SOURCE_FILE, PAGE_NUM)

    assert_page_matches_results(
        page,
        source_file=SOURCE_FILE,
        page_num=PAGE_NUM,
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
        "embedding": [0.25] * 768,
    }
    updated_hash = "hash-2"

    save_example_page(
        conn,
        content_hash=updated_hash,
        results=updated_results,
    )

    page = db.get_page(conn, SOURCE_FILE, PAGE_NUM)
    assert_page_matches_results(
        page,
        source_file=SOURCE_FILE,
        page_num=PAGE_NUM,
        content_hash=updated_hash,
        results=updated_results,
    )


def test_get_hashes_returns_hashes_keyed_by_source_file_and_page_number(conn):
    save_example_page(conn, source_file="lecture-a.pdf", page_num=1, content_hash="a1")
    save_example_page(conn, source_file="lecture-a.pdf", page_num=2, content_hash="a2")
    save_example_page(conn, source_file="lecture-b.pdf", page_num=1, content_hash="b1")

    assert db.get_hashes(conn) == {
        ("lecture-a.pdf", 1): "a1",
        ("lecture-a.pdf", 2): "a2",
        ("lecture-b.pdf", 1): "b1",
    }


def test_get_hashes_returns_empty_dict_when_table_is_empty(conn):
    assert db.get_hashes(conn) == {}


@pytest.mark.parametrize(
    ("field", "empty_value"),
    [
        ("content_hash", ""),
        ("transcript", ""),
        ("concepts", []),
        ("summary", ""),
        ("prompt_version", ""),
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
            SOURCE_FILE,
            PAGE_NUM,
            results.pop("content_hash", CONTENT_HASH),
            **results,
        )

    assert db.get_page(conn, SOURCE_FILE, PAGE_NUM) is None


def test_pgvector_conversion_round_trips_float_values():
    vector = [0.0, -1.25, 3.5]

    encoded = db.to_pgvector(vector)

    assert encoded == "[0.0,-1.25,3.5]"
    assert db.from_pgvector(encoded) == pytest.approx(vector)
