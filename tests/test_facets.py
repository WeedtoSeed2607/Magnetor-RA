"""Branch C — facet classification: the mode of approach, not the subject."""

from __future__ import annotations

import datetime as dt

from magnetor.facets import (
    CONCEPTUAL,
    EMPIRICAL,
    FORMAL,
    NORMATIVE,
    UNCLASSIFIED,
    classify,
    classify_text,
    cross_facet_neighbours,
    facet_counts,
    node_facets,
)
from magnetor.harvest import HarvestedPaper, HarvestResult


def _paper(wid: str, title: str, abstract: str = "") -> HarvestedPaper:
    return HarvestedPaper(
        openalex_id=wid, title=title, year=2000, doi=None, venue=None,
        cited_by_count=0, referenced_works=(), institution_countries=(),
        is_review=False, is_retracted=False, abstract=abstract,
    )


def test_empirical_and_formal_are_both_assignable() -> None:
    """Multi-label by design: a paper can measure and prove without contradiction."""
    facets, evidence = classify_text(
        "A study of decoders",
        "We ran an experiment with 40 participants and prove a theorem bounding "
        "the algorithm's complexity.",
    )
    assert EMPIRICAL in facets
    assert FORMAL in facets
    assert set(evidence[FORMAL]) >= {"theorem", "complexity"}


def test_strongest_evidence_is_listed_first() -> None:
    facets, evidence = classify_text(
        "Title",
        "theorem proof lemma axiom equation. We also observed one measurement.",
    )
    assert facets[0] == FORMAL
    assert len(evidence[FORMAL]) > len(evidence.get(EMPIRICAL, ()))


def test_a_single_term_is_not_enough() -> None:
    """One match is a coincidence; the bar is two distinct terms."""
    assert classify_text("An experiment", "") == ((UNCLASSIFIED,), {})
    facets, _ = classify_text("An experiment", "with measurement")
    assert EMPIRICAL in facets


def test_unclassified_is_a_first_class_outcome() -> None:
    facets, evidence = classify_text("On some things", "A short note about things.")
    assert facets == (UNCLASSIFIED,)
    assert evidence == {}


def test_evidence_terms_make_an_assignment_checkable() -> None:
    title = "Defining morality"
    abstract = "A conceptual analysis offering a definition and a distinction."
    _facets, evidence = classify_text(title, abstract)
    assert CONCEPTUAL in evidence
    # Every recorded term must be findable in what was actually read — title and
    # abstract both, since the title carries method words too ("Defining").
    source = f"{title} {abstract}".lower()
    for term in evidence[CONCEPTUAL]:
        assert term in source


def test_min_terms_is_tunable() -> None:
    strict = classify_text("Optimal foraging", "adaptive", min_terms=3)[0]
    loose = classify_text("Optimal foraging", "adaptive", min_terms=2)[0]
    assert strict == (UNCLASSIFIED,)
    assert NORMATIVE in loose


def test_classify_covers_every_harvested_paper() -> None:
    result = HarvestResult(
        query="q", generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(),
        papers=(
            _paper("A", "Experimental study", "experiment measurement observed"),
            _paper("B", "Nothing in particular", ""),
        ),
        edges=(), n_fetched=2,
    )
    index = classify(result)
    assert {a.openalex_id for a in index.assignments} == {"A", "B"}
    assert index.by_id()["B"].facets == (UNCLASSIFIED,)


def test_facet_counts_are_multi_label() -> None:
    nodes: list[dict[str, object]] = [
        {"id": "A", "facets": ["empirical", "formal"]},
        {"id": "B", "facets": ["empirical"]},
        {"id": "C"},  # no facets recorded -> unclassified
    ]
    counts = facet_counts(nodes)
    assert counts["empirical"] == 2
    assert counts["formal"] == 1
    assert counts[UNCLASSIFIED] == 1
    assert sum(counts.values()) > len(nodes)  # multi-label oversums, by design


def test_node_facets_defaults_to_unclassified() -> None:
    assert node_facets({"id": "A"}) == (UNCLASSIFIED,)
    assert node_facets({"id": "A", "facets": []}) == (UNCLASSIFIED,)
    assert node_facets({"id": "A", "facets": ["formal"]}) == ("formal",)


def test_cross_facet_neighbours_surface_a_different_approach() -> None:
    """The concern-1 payoff: same foundations, different mode of attack."""
    nodes: dict[str, dict[str, object]] = {
        "ME": {"id": "ME", "facets": ["empirical"]},
        "SAME": {"id": "SAME", "facets": ["empirical"]},
        "OTHER": {"id": "OTHER", "facets": ["formal"]},
        "BLANK": {"id": "BLANK", "facets": ["unclassified"]},
    }
    found = cross_facet_neighbours("ME", [("SAME", 9), ("OTHER", 4), ("BLANK", 7)], nodes)
    assert [nid for nid, _w, _f in found] == ["OTHER"]
    assert found[0][2] == ("formal",)


def test_cross_facet_ignores_unclassified_partners() -> None:
    nodes: dict[str, dict[str, object]] = {
        "ME": {"id": "ME", "facets": ["formal"]}, "X": {"id": "X"},
    }
    assert cross_facet_neighbours("ME", [("X", 5)], nodes) == ()
