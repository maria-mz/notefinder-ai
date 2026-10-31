import pytest

from notefinder.db import PageId
from notefinder.search import Bm25Index, search, tokenize
from tests.helpers import MockEmbedder, axis_vector, save_test_page


def page(source_file: str) -> PageId:
    return PageId(source_file, 1)


def test_search_finds_the_page_nearest_to_the_query(conn):
    save_test_page(conn, page("logic.pdf"), embedding=axis_vector(1, 0))
    save_test_page(conn, page("graphs.pdf"), embedding=axis_vector(0, 1))
    vectors = {"logic": axis_vector(1, 0), "graphs": axis_vector(0, 1)}
    embedder = MockEmbedder(lambda query: vectors[query])

    assert search(conn, embedder, "graphs", k=1)[0].id == page("graphs.pdf")
    assert search(conn, embedder, "logic", k=1)[0].id == page("logic.pdf")


def test_search_returns_k_results(conn):
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        save_test_page(conn, page(name), embedding=axis_vector(1, 0))
    embedder = MockEmbedder(lambda query: axis_vector(1, 0))

    assert len(search(conn, embedder, "topic", k=2)) == 2


def test_search_only_finds_pages_embedded_with_the_embedders_model(conn):
    save_test_page(conn, page("current.pdf"), embed_model="model-b")
    save_test_page(conn, page("old.pdf"), embed_model="model-a")
    embedder = MockEmbedder(lambda query: axis_vector(1, 0), model_name="model-b")

    assert [r.id for r in search(conn, embedder, "topic", k=5)] == [page("current.pdf")]


A, B, C, D, E = (PageId("notes.pdf", n) for n in range(1, 6))


@pytest.mark.parametrize("text, tokens", [
    ("Hall's theorem", ["hall", "s", "theorem"]),
    ("big-O", ["big", "o"]),
    ("∀x P(x) → Q", ["x", "p", "x", "q"]),
    ("snake_case 2nd", ["snake", "case", "2nd"]),
    ("", []),
])
def test_tokenize_splits_into_lowercase_runs_of_letters_and_digits(text, tokens):
    assert tokenize(text) == tokens


def test_tokenize_stems_words():
    tokens = tokenize("relations Relation trees diagrams")

    assert tokens == ["relat", "relat", "tree", "diagram"]


def test_rank_matches_other_forms_of_a_word():
    index = Bm25Index({A: "a relation on a set", B: "binary trees", C: "truth table"})

    assert index.rank("mathematical relations") == [A]


def test_rank_weighs_rare_words_over_common_ones():
    index = Bm25Index({
        A: "probability of a die roll",
        B: "probability tables",
        C: "probability trees",
        D: "conditional probability",
    })

    assert index.rank("die probability")[0] == A


def test_rank_leaves_out_pages_without_a_query_word():
    index = Bm25Index({A: "hasse diagram", B: "truth table", C: "venn diagram"})

    assert set(index.rank("hasse diagram")) == {A, C}
    assert index.rank("nothing matches") == []


def test_rank_breaks_ties_by_page():
    # BM25 gives a word on exactly half the pages zero weight, so the pages
    # without "heap" are there to keep it under half
    index = Bm25Index({C: "heap", A: "heap", B: "stack", D: "queue", E: "tree"})

    assert index.rank("heap") == [A, C]
