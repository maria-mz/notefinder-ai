from notefinder.db import PageId
from notefinder.search import search
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
