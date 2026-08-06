"""Branch C · L4 — derived relations: bibliographic coupling and co-citation."""

from __future__ import annotations

from magnetor.relations import (
    DEFAULT_MAX_FANOUT,
    RelationEdge,
    bibliographic_coupling,
    co_citation,
)


def _refs(**mapping: str) -> dict[str, frozenset[str]]:
    """``a="X Y Z"`` -> ``{"a": {"X","Y","Z"}}``."""
    return {key: frozenset(value.split()) for key, value in mapping.items()}


def test_coupling_counts_shared_references() -> None:
    # A and B share three references; neither cites the other.
    refs = _refs(A="X Y Z Q", B="X Y Z R", C="M N")
    edges = bibliographic_coupling(refs, min_shared=3)
    assert edges == (RelationEdge(source="A", target="B", weight=3),)


def test_coupling_counts_references_outside_the_harvested_set() -> None:
    """The point of the layer: shared *external* ancestry still links two papers.

    X, Y and Z are not themselves harvested papers, so nothing about this pair is
    visible in the citation backbone.
    """
    refs = _refs(A="X Y Z", B="X Y Z")
    edges = bibliographic_coupling(refs, min_shared=3)
    assert len(edges) == 1
    assert set(refs) & {"X", "Y", "Z"} == set()  # the shared refs are out-of-set


def test_coupling_respects_the_minimum() -> None:
    refs = _refs(A="X Y", B="X Y")
    assert bibliographic_coupling(refs, min_shared=3) == ()
    assert len(bibliographic_coupling(refs, min_shared=2)) == 1


def test_coupling_is_undirected_and_canonically_ordered() -> None:
    edges = bibliographic_coupling(_refs(zeta="X Y Z", alpha="X Y Z"), min_shared=3)
    assert edges[0].source == "alpha" and edges[0].target == "zeta"


def test_high_fanout_reference_is_skipped() -> None:
    """A reference everyone cites pairs everyone with everyone and says nothing."""
    refs = {name: frozenset({"STAPLE"}) for name in "abcdef"}
    assert co_citation(refs, min_co_citations=1, max_fanout=3) == ()
    assert bibliographic_coupling(refs, min_shared=1, max_fanout=3) == ()
    assert bibliographic_coupling(refs, min_shared=1, max_fanout=DEFAULT_MAX_FANOUT)


def test_co_citation_counts_only_pairs_that_are_both_harvested() -> None:
    # C and D each cite both A and B; A and B are harvested, OUTSIDE is not.
    refs = _refs(A="", B="", C="A B OUTSIDE", D="A B OUTSIDE")
    edges = co_citation(refs, min_co_citations=2)
    assert edges == (RelationEdge(source="A", target="B", weight=2),)


def test_co_citation_requires_repetition() -> None:
    refs = _refs(A="", B="", C="A B")
    assert co_citation(refs, min_co_citations=2) == ()
    assert len(co_citation(refs, min_co_citations=1)) == 1


def test_top_k_caps_edges_per_node_keeping_the_strongest() -> None:
    # HUB shares many refs with P1..P4; keeping k=2 must retain the heaviest.
    refs = {
        "HUB": frozenset({"r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"}),
        "P1": frozenset({"r1", "r2", "r3", "r4"}),  # weight 4 — strongest
        "P2": frozenset({"r1", "r2", "r3"}),  # weight 3
        "P3": frozenset({"r5", "r6"}),  # weight 2 — below min_shared
        "P4": frozenset({"r7", "r8"}),  # weight 2 — below min_shared
    }
    edges = bibliographic_coupling(refs, min_shared=3, top_k=2)
    touching_hub = [e for e in edges if "HUB" in (e.source, e.target)]
    partners = {e.target if e.source == "HUB" else e.source for e in touching_hub}
    assert partners == {"P1", "P2"}
    assert all(e.weight >= 3 for e in edges)


def test_edges_are_sorted_by_weight_descending() -> None:
    refs = {
        "A": frozenset({"1", "2", "3", "4"}),
        "B": frozenset({"1", "2", "3", "4"}),  # weight 4 with A
        "C": frozenset({"1", "2", "3"}),  # weight 3 with A and B
    }
    weights = [e.weight for e in bibliographic_coupling(refs, min_shared=3)]
    assert weights == sorted(weights, reverse=True)


def test_empty_input_is_handled() -> None:
    assert bibliographic_coupling({}) == ()
    assert co_citation({}) == ()
