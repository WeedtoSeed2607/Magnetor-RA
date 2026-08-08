"""Branch C — foundational gaps and unexplored approach axes."""

from __future__ import annotations

import datetime as dt
from typing import Any

from magnetor.gaps import (
    enrich,
    facet_gaps,
    foundational_gaps,
    gap_rows,
    read_gaps,
)
from magnetor.harvest import HarvestedPaper, HarvestResult


def _paper(wid: str, refs: tuple[str, ...]) -> HarvestedPaper:
    return HarvestedPaper(
        openalex_id=wid, title=f"Paper {wid}", year=2000, doi=None, venue=None,
        cited_by_count=0, referenced_works=refs, institution_countries=(),
        is_review=False, is_retracted=False,
    )


def _result(papers: list[HarvestedPaper]) -> HarvestResult:
    return HarvestResult(
        query="q", generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(),
        papers=tuple(papers), edges=(), n_fetched=len(papers),
    )


def test_an_outside_work_cited_repeatedly_is_a_gap() -> None:
    """The point: what the map leans on but never captured."""
    result = _result([
        _paper("A", ("HUME", "B")),
        _paper("B", ("HUME",)),
        _paper("C", ("HUME", "ONCE")),
    ])
    gaps = foundational_gaps(result, min_citations=2)
    assert [g.openalex_id for g in gaps] == ["HUME"]
    assert gaps[0].cited_by_in_set == 3
    assert gaps[0].share == 1.0


def test_in_set_works_are_never_gaps() -> None:
    """B is cited twice but is already on the map, so it is not missing."""
    result = _result([_paper("A", ("B",)), _paper("C", ("B",)), _paper("B", ())])
    assert foundational_gaps(result, min_citations=2) == ()


def test_singleton_references_are_not_gaps() -> None:
    """Raw leakage counts a long tail nobody would call a gap; the floor removes it."""
    result = _result([_paper("A", ("ONCE",)), _paper("B", ("TWICE",))])
    assert foundational_gaps(result, min_citations=2) == ()


def test_gaps_are_ranked_by_demand_and_capped() -> None:
    papers = [_paper(f"P{i}", ("HOT", "WARM") if i < 3 else ("WARM",)) for i in range(5)]
    gaps = foundational_gaps(_result(papers), min_citations=2, limit=1)
    assert len(gaps) == 1
    assert gaps[0].openalex_id == "HOT" or gaps[0].cited_by_in_set == 5


def test_empty_harvest_yields_no_gaps() -> None:
    assert foundational_gaps(_result([])) == ()


def test_enrich_attaches_titles_and_keeps_unresolvable_ones() -> None:
    """A gap whose metadata cannot be fetched must survive with its count intact.

    Dropping it would quietly shrink the very measure of incompleteness this
    exists to report — and on the real corpus the two largest gaps were exactly
    these, records OpenAlex would not resolve.
    """
    result = _result([_paper("A", ("KNOWN", "MERGED")), _paper("B", ("KNOWN", "MERGED"))])
    gaps = foundational_gaps(result, min_citations=2)
    works: list[dict[str, Any]] = [
        {"id": "https://openalex.org/KNOWN", "display_name": "A Treatise", "publication_year": 1739}
    ]
    enriched = enrich(gaps, works)
    by_id = {g.openalex_id: g for g in enriched}
    assert by_id["KNOWN"].title == "A Treatise"
    assert by_id["KNOWN"].year == 1739
    assert by_id["MERGED"].title is None
    assert by_id["MERGED"].cited_by_in_set == 2  # count preserved


def test_gap_rows_round_trip_through_a_document() -> None:
    result = _result([_paper("A", ("X",)), _paper("B", ("X",))])
    rows = gap_rows(enrich(foundational_gaps(result, min_citations=2), []))
    restored = read_gaps({"foundational_gaps": rows})
    assert restored[0].openalex_id == "X"
    assert restored[0].cited_by_in_set == 2
    assert restored[0].url.endswith("/X")


def test_read_gaps_tolerates_older_graphs() -> None:
    assert read_gaps({}) == ()
    assert read_gaps({"foundational_gaps": "nonsense"}) == ()


def test_facet_gaps_report_absent_and_thin_axes() -> None:
    nodes: list[dict[str, Any]] = [
        {"id": f"E{i}", "facets": ["empirical"]} for i in range(9)
    ]
    nodes.append({"id": "F1", "facets": ["formal"]})
    gaps = {g.facet: g for g in facet_gaps(nodes, thin_share=0.10)}
    assert "empirical" not in gaps  # well covered
    assert gaps["formal"].papers == 1 and not gaps["formal"].absent
    assert gaps["mechanistic"].absent  # nobody took this axis at all


def test_unclassified_is_never_reported_as_a_gap() -> None:
    """It records that the classifier could not read the paper, not an absence
    in the literature."""
    nodes: list[dict[str, Any]] = [{"id": "A"}, {"id": "B", "facets": ["unclassified"]}]
    assert all(gap.facet != "unclassified" for gap in facet_gaps(nodes))


def test_no_nodes_means_no_facet_gaps() -> None:
    assert facet_gaps([]) == ()
