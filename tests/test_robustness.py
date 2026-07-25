"""Branch C · L8 robustness: bootstrap rank CIs, boundary leakage, Kendall tau."""

from __future__ import annotations

import datetime as dt

from magnetor.harvest import HarvestedPaper, HarvestResult
from magnetor.robustness import bootstrap_rank_cis, boundary_leakage, kendall_tau


def _paper(wid: str, refs: tuple[str, ...] = ()) -> HarvestedPaper:
    return HarvestedPaper(
        openalex_id=wid, title=f"P{wid}", year=2020, doi=None, venue=None,
        cited_by_count=0, referenced_works=refs, institution_countries=(),
        is_review=False, is_retracted=False,
    )


def _result(papers: list[HarvestedPaper], edges: list[tuple[str, str]]) -> HarvestResult:
    return HarvestResult(
        query="q", generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(),
        papers=tuple(papers), edges=tuple(edges), n_fetched=len(papers),
    )


def test_boundary_leakage_fraction() -> None:
    # W1 cites W2 (internal) and X, Y (external) -> 2/3 leak.
    result = _result([_paper("W1", ("W2", "X", "Y")), _paper("W2")], [("W1", "W2")])
    assert boundary_leakage(result) == 2 / 3


def test_boundary_leakage_zero_when_no_refs() -> None:
    assert boundary_leakage(_result([_paper("W1"), _paper("W2")], [])) == 0.0


def test_bootstrap_ranks_put_central_paper_first() -> None:
    # A is cited by B and C; it should hold the best (lowest) median rank.
    papers = [_paper("A"), _paper("B", ("A",)), _paper("C", ("A", "B"))]
    edges = [("B", "A"), ("C", "A"), ("C", "B")]
    rob = bootstrap_rank_cis(_result(papers, edges), resamples=200, seed=1)
    assert rob.intervals[0].openalex_id == "A"
    assert rob.intervals[0].median_rank <= rob.intervals[-1].median_rank
    assert rob.resamples == 200


def test_bootstrap_intervals_are_ordered_and_bounded() -> None:
    papers = [_paper("A"), _paper("B", ("A",)), _paper("C", ("A", "B"))]
    rob = bootstrap_rank_cis(_result(papers, [("B", "A"), ("C", "A")]), resamples=100, seed=2)
    medians = [r.median_rank for r in rob.intervals]
    assert medians == sorted(medians)  # ordered best-first
    for r in rob.intervals:
        assert r.lo_rank <= r.hi_rank
        assert r.lo_rank >= 1


def test_bootstrap_is_deterministic_under_seed() -> None:
    papers = [_paper("A"), _paper("B", ("A",)), _paper("C", ("A",))]
    r = _result(papers, [("B", "A"), ("C", "A")])
    a = bootstrap_rank_cis(r, resamples=100, seed=7)
    b = bootstrap_rank_cis(r, resamples=100, seed=7)
    assert [i.openalex_id for i in a.intervals] == [i.openalex_id for i in b.intervals]
    assert [i.median_rank for i in a.intervals] == [i.median_rank for i in b.intervals]


def test_empty_graph_is_robustly_empty() -> None:
    rob = bootstrap_rank_cis(_result([], []), resamples=10)
    assert rob.intervals == ()
    assert rob.boundary_leakage == 0.0


def test_kendall_tau_identical_and_reversed() -> None:
    order = ["a", "b", "c", "d"]
    assert kendall_tau(order, order) == 1.0
    assert kendall_tau(order, list(reversed(order))) == -1.0


def test_kendall_tau_partial_agreement_between_bounds() -> None:
    tau = kendall_tau(["a", "b", "c", "d"], ["a", "c", "b", "d"])  # one swap
    assert -1.0 < tau < 1.0
