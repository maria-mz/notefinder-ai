import pytest

from notefinder.db import PageId
from notefinder.evaluate import (
    LabeledQuery, LabelError, QueryResult, SearchMethod, check_labels, evaluate,
    load_labels, mean_recall, parse_labels, recall_at_k, run_record,
)
from tests.helpers import PARAMS

A, B, C, D, E, F, G = (PageId("notes.pdf", n) for n in range(1, 8))


def write_labels(tmp_path, text: str):
    path = tmp_path / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "ranked, relevant, expected",
    [
        ([A, B, C], {A}, 1.0),
        ([B, C, A], {A}, 0.0),               # just outside the top 2
        ([A, C, B], {A, B}, 0.5),
        # 7 relevant: scored out of 2, not 7
        ([A, B, C, D, E, F, G], {A, B, C, D, E, F, G}, 1.0),
        ([A, F, B, G], {A, B, C, D}, 0.5),
    ],
)
def test_recall_at_k(ranked, relevant, expected):
    assert recall_at_k(ranked, frozenset(relevant), k=2) == expected


def test_evaluate_scores_each_query_and_ranks_its_relevant_pages():
    rankings = {"first": [A, B, C, D], "second": [C, D, A, B]}
    queries = [
        LabeledQuery("first", frozenset({A, D})),
        LabeledQuery("second", frozenset({B, E})),  # E isn't ranked at all
    ]

    results = evaluate(queries, rankings.__getitem__, k=2)

    assert results == [
        QueryResult("first", recall=0.5, ranks={A: 1, D: 4}),
        QueryResult("second", recall=0.0, ranks={B: 4, E: None}),
    ]
    assert mean_recall(results) == 0.25


def test_missed_lists_pages_outside_the_top_k_in_page_order():
    result = QueryResult("q", recall=0.5, ranks={C: 9, A: 1, B: None})

    assert result.missed(k=5) == [(B, None), (C, 9)]


def test_load_labels(tmp_path):
    path = write_labels(tmp_path, """
relevance_rule: "anything"
queries:
  - query: heaps
    relevant:
      - {file: dsa/a.pdf, page: 2}
      - {file: dsa/b.pdf, page: 1}
  - query: DP
    relevant:
      - {file: dsa/a.pdf, page: 5}
""")

    assert load_labels(path) == [
        LabeledQuery("heaps",
                     frozenset({PageId("dsa/a.pdf", 2), PageId("dsa/b.pdf", 1)})),
        LabeledQuery("DP", frozenset({PageId("dsa/a.pdf", 5)})),
    ]


PAGE_1 = {"file": "a.pdf", "page": 1}


@pytest.mark.parametrize(
    "data, message",
    [
        (None, "'queries' list"),  # an empty file
        ({"relevance_rule": "x"}, "'queries' list"),
        ({"queries": [{"relevant": [PAGE_1]}]}, "no 'query' text"),
        ({"queries": [{"query": "  ", "relevant": [PAGE_1]}]}, "no 'query' text"),
        ({"queries": [{"query": "heaps", "relevant": []}]}, "no relevant pages"),
        ({"queries": [{"query": "heaps", "relevant": [{"file": "a.pdf"}]}]},
         "isn't {file, page}"),
        ({"queries": [{"query": "heaps", "relevant": [{"file": "a.pdf", "page": "1"}]}]},
         "isn't {file, page}"),
        ({"queries": [{"query": "heaps", "relevant": [PAGE_1]},
                      {"query": "heaps", "relevant": [{"file": "a.pdf", "page": 2}]}]},
         "duplicate query"),
        ({"queries": [{"query": "heaps", "relevant": [PAGE_1, PAGE_1]}]},
         "lists a page twice"),
    ],
)
def test_parse_labels_rejects_malformed_labels(data, message):
    with pytest.raises(LabelError, match=message):
        parse_labels(data)


def test_check_labels_passes_when_every_labelled_page_is_stored():
    queries = [LabeledQuery("q", frozenset({A}))]

    assert check_labels(queries, {A, B}) is None


def test_check_labels_lists_missing_labelled_pages():
    queries = [LabeledQuery("q", frozenset({A, B, C}))]

    problem = check_labels(queries, {A})

    assert problem.startswith("2 labelled pages aren't in the database")
    assert "notes.pdf p.2" in problem and "notes.pdf p.3" in problem


def test_run_record():
    results = [
        QueryResult("first", recall=1.0, ranks={B: 2, A: 1}),
        QueryResult("second", recall=0.0, ranks={C: None}),
    ]

    record = run_record(results, k=5, method=SearchMethod.BM25, params=PARAMS,
                        num_pages=87, timestamp="2026-10-30T12:00:00+00:00")

    assert record == {
        "timestamp": "2026-10-30T12:00:00+00:00",
        "method": "bm25",
        "prompt_version": PARAMS.prompt.version,
        "claude_model": PARAMS.claude_model,
        "embed_model": PARAMS.embed_model,
        "k": 5,
        "num_pages": 87,
        "num_queries": 2,
        "mean_recall": 0.5,
        "queries": [
            {"query": "first", "recall": 1.0, "relevant": [
                {"file": "notes.pdf", "page": 1, "rank": 1},
                {"file": "notes.pdf", "page": 2, "rank": 2},
            ]},
            {"query": "second", "recall": 0.0, "relevant": [
                {"file": "notes.pdf", "page": 3, "rank": None},
            ]},
        ],
    }
