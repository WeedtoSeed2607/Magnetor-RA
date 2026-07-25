"""Branch C · L3.0 scoring: in-degree, PageRank, percentile normalisation."""

from __future__ import annotations

import datetime as dt

from magnetor.graph_scoring import score_graph
from magnetor.harvest import HarvestedPaper, HarvestResult


def _paper(wid: str) -> HarvestedPaper:
    return HarvestedPaper(
        openalex_id=wid, title=f"P{wid}", year=2020, doi=None, venue=None,
        cited_by_count=0, referenced_works=(), institution_countries=(),
        is_review=False, is_retracted=False,
    )


def _result(ids: list[str], edges: list[tuple[str, str]]) -> HarvestResult:
    return HarvestResult(
        query="q", generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(),
        papers=tuple(_paper(i) for i in ids), edges=tuple(edges), n_fetched=len(ids),
    )


def test_empty_graph_scores_nothing() -> None:
    assert score_graph(_result([], [])).scored == ()


def test_in_degree_counts_internal_citations() -> None:
    # B->A, C->A, C->B  => A cited twice, B once, C never.
    scores = score_graph(_result(["A", "B", "C"], [("B", "A"), ("C", "A"), ("C", "B")]))
    by_id = {s.openalex_id: s for s in scores.scored}
    assert by_id["A"].in_degree == 2
    assert by_id["B"].in_degree == 1
    assert by_id["C"].in_degree == 0


def test_most_cited_paper_ranks_first() -> None:
    scores = score_graph(_result(["A", "B", "C"], [("B", "A"), ("C", "A"), ("C", "B")]))
    # A receives the most in-set citations -> highest PageRank -> first.
    assert scores.scored[0].openalex_id == "A"
    assert scores.scored[0].influence == 1.0  # top percentile


def test_pagerank_is_a_distribution() -> None:
    scores = score_graph(_result(["A", "B", "C"], [("B", "A"), ("C", "A"), ("C", "B")]))
    total = sum(s.pagerank for s in scores.scored)
    assert abs(total - 1.0) < 1e-6  # PageRank mass sums to 1


def test_percentiles_bounded_and_monotone_with_indegree() -> None:
    scores = score_graph(_result(["A", "B", "C"], [("B", "A"), ("C", "A"), ("C", "B")]))
    for s in scores.scored:
        assert 0.0 < s.in_degree_pct <= 1.0
        assert 0.0 < s.pagerank_pct <= 1.0
    by_id = {s.openalex_id: s for s in scores.scored}
    # Higher in-degree => higher-or-equal in-degree percentile.
    assert by_id["A"].in_degree_pct >= by_id["B"].in_degree_pct >= by_id["C"].in_degree_pct


def test_isolated_nodes_share_bottom_percentile() -> None:
    # No edges: every node ties at in-degree 0 -> all percentile 1.0 (all <= each).
    scores = score_graph(_result(["A", "B"], []))
    assert all(s.in_degree == 0 for s in scores.scored)
    assert all(s.in_degree_pct == 1.0 for s in scores.scored)
