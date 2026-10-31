"""Scoring search against hand-labelled queries: mean recall@k."""
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Callable

import yaml

from notefinder.db import PageId
from notefinder.sync.plan import SyncParams


#: How many missing labelled pages :func:`check_labels` names before summarizing.
MAX_MISSING_SHOWN = 10


class SearchMethod(StrEnum):
    """What ranks the pages for each query."""
    # BGE embedding
    EMBEDDING = "embedding"
    
    # keyword search over transcripts
    BM25 = "bm25"

    # keyword search over embedded text
    BM25_EMBEDDED_TEXT = "bm25-embedded-text" 


class LabelError(Exception):
    """The labels are malformed."""


@dataclass(frozen=True)
class LabeledQuery:
    """A query and the pages a perfect search would return for it.

    :ivar query: The search text.
    :ivar relevant: Every relevant page. Never empty.
    """
    query: str
    relevant: frozenset[PageId]


@dataclass
class QueryResult:
    """How one query scored.

    :ivar query: The search text.
    :ivar recall: Its recall@k.
    :ivar ranks: Each relevant page's 1-based rank, or ``None`` if it wasn't ranked.
    """
    query: str
    recall: float
    ranks: dict[PageId, int | None]

    def missed(self, k: int) -> list[tuple[PageId, int | None]]:
        """List the relevant pages outside the top k.

        :param k: The cutoff.
        :returns: Each missed page and its rank, in page order.
        """
        return sorted((p, r) for p, r in self.ranks.items() if r is None or r > k)


def parse_page(entry, query: str) -> PageId:
    """Parse one relevant page of a labelled query.

    :param entry: The page as read from YAML, expected to be ``{file, page}``.
    :param query: The query it's labelled for, to name in errors.
    :returns: The page.
    :raises LabelError: If ``entry`` isn't a ``{file, page}`` mapping.
    """
    if not (isinstance(entry, dict)
            and isinstance(entry.get("file"), str)
            and isinstance(entry.get("page"), int)):
        raise LabelError(f"{query!r} has a page that isn't {{file, page}}: {entry!r}")
    return PageId(entry["file"], entry["page"])


def load_labels(path: Path) -> list[LabeledQuery]:
    """Read a labels file, and check and convert it with :func:`parse_labels`.

    :param path: The YAML file.
    :returns: The labelled queries, in file order.
    :raises OSError: If the file can't be read.
    :raises yaml.YAMLError: If it isn't valid YAML.
    :raises LabelError: If :func:`parse_labels` rejects it.
    """
    return parse_labels(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_labels(data) -> list[LabeledQuery]:
    """Check labels as read from YAML, and convert them.

    :param data: Expected to be a mapping with a ``queries`` list of
        ``{query, relevant: [{file, page}, ...]}``.
    :returns: The labelled queries, in order.
    :raises LabelError: If ``data`` doesn't have that shape, or repeats a query
        or a query's page.
    """
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise LabelError("expected a top-level 'queries' list")

    queries: list[LabeledQuery] = []
    seen: set[str] = set()

    for i, item in enumerate(data["queries"], start=1):
        query = item.get("query") if isinstance(item, dict) else None
        if not (isinstance(query, str) and query.strip()):
            raise LabelError(f"query #{i} has no 'query' text")
        if query in seen:
            raise LabelError(f"duplicate query: {query!r}")
        seen.add(query)

        relevant = item.get("relevant")
        if not (isinstance(relevant, list) and relevant):
            raise LabelError(f"{query!r} has no relevant pages")

        pages = [parse_page(entry, query) for entry in relevant]
        if len(set(pages)) != len(pages):
            raise LabelError(f"{query!r} lists a page twice")

        queries.append(LabeledQuery(query, frozenset(pages)))

    return queries


def recall_at_k(ranked: list[PageId], relevant: frozenset[PageId], k: int) -> float:
    """Score a ranking by how many relevant pages it puts in the top k.

    Dividing by min(|relevant|, k) rather than |relevant| lets a query with more
    than k relevant pages still score 1.0.

    :param ranked: The pages, best first.
    :param relevant: The pages that should rank high. Must not be empty.
    :param k: The cutoff.
    :returns: The share of relevant pages in the top k, out of at most k: 0 to 1.
    """
    found = len(relevant & set(ranked[:k]))
    return found / min(len(relevant), k)


def evaluate(queries: list[LabeledQuery], rank: Callable[[str], list[PageId]],
             k: int) -> list[QueryResult]:
    """Score each query.

    :param queries: The labelled queries.
    :param rank: Returns pages for a query, best first.
    :param k: The cutoff.
    :returns: One result per query, in order.
    """
    results = []
    for q in queries:
        ranked = rank(q.query)
        positions = {page_id: i for i, page_id in enumerate(ranked, start=1)}
        results.append(QueryResult(
            query=q.query,
            recall=recall_at_k(ranked, q.relevant, k),
            ranks={p: positions.get(p) for p in q.relevant},
        ))
    return results


def mean_recall(results: list[QueryResult]) -> float:
    """Average the queries' recall.

    :param results: The results. Must not be empty.
    :returns: The mean recall@k.
    """
    return sum(r.recall for r in results) / len(results)


def check_labels(queries: list[LabeledQuery], stored: set[PageId]) -> str | None:
    """Check that every labelled page is stored.

    :param queries: The labelled queries.
    :param stored: Every stored page.
    :returns: Which labelled pages are missing, or ``None`` if none are.
    """
    missing = sorted({p for q in queries for p in q.relevant} - stored)
    if not missing:
        return None
    shown = ", ".join(str(p) for p in missing[:MAX_MISSING_SHOWN])
    more = len(missing) - MAX_MISSING_SHOWN
    return (f"{len(missing)} labelled pages aren't in the database: {shown}"
            + (f", and {more} more" if more > 0 else ""))


def run_record(results: list[QueryResult], *, k: int, method: SearchMethod,
               params: SyncParams, num_pages: int, timestamp: str) -> dict:
    """Gather everything needed to compare this run with later ones.

    :param results: The results. Must not be empty.
    :param k: The cutoff they were scored at.
    :param method: What ranked the pages.
    :param params: Supplies the prompt version and models the pages were made with.
    :param num_pages: How many pages were ranked.
    :param timestamp: When the run happened, as ISO 8601.
    :returns: The record, as JSON-ready data.
    """
    return {
        "timestamp": timestamp,
        "method": str(method),
        "prompt_version": params.prompt.version,
        "claude_model": params.claude_model,
        "embed_model": params.embed_model,
        "k": k,
        "num_pages": num_pages,
        "num_queries": len(results),
        "mean_recall": mean_recall(results),
        "queries": [
            {
                "query": r.query,
                "recall": r.recall,
                "relevant": [
                    {"file": p.source_file, "page": p.page_num, "rank": rank}
                    for p, rank in sorted(r.ranks.items())
                ],
            }
            for r in results
        ],
    }


def save_run(record: dict, runs_dir: Path, now: datetime) -> Path:
    """Save an eval run's record as JSON, named by when it ran.

    :param record: The record to save.
    :param runs_dir: The folder to save it in. Created if missing.
    :param now: When the run happened.
    :returns: The path it was saved to.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{now:%Y-%m-%dT%H-%M-%S}.json"
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
