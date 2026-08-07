"""Branch C · L5.2/L5.3: relevance seeding, highlighting, and progression paths."""

from __future__ import annotations

from typing import Any

from magnetor.pathways import (
    DEVELOPMENT,
    INDIRECT,
    LINEAGE,
    ROOTS,
    UNCONNECTED,
    choose_anchor,
    connect,
    graph_view,
    highlight,
    path_edges,
    personalised_pagerank,
    progression_paths,
    relevance_vector,
    step_weights,
)

#: Matches MID alone above the threshold, so the anchor lands mid-chain and both
#: legs have somewhere to go. ("surface code…" also matches OLD, which puts the
#: anchor at the very start of the lineage — covered separately.)
_MID_QUERY = "decoders improved"
_MID_THRESHOLD = 0.6


def _node(nid: str, title: str, year: int | None, influence: float) -> dict[str, Any]:
    return {"id": nid, "title": title, "year": year, "influence": influence}


def _doc() -> dict[str, Any]:
    """A small lineage: OLD <- MID <- NEW, plus an unrelated island.

    Edges point citing -> cited, so NEW cites MID cites OLD. "surface code" is
    the topical thread; DECOY shares no vocabulary with it.
    """
    return {
        "nodes": [
            _node("OLD", "Surface code foundations", 1996, 0.9),
            _node("MID", "Surface code decoders improved", 2010, 0.6),
            _node("NEW", "Surface code decoders at scale", 2022, 0.3),
            _node("DECOY", "Protein folding kinetics", 2001, 0.5),
        ],
        "edges": [["MID", "OLD"], ["NEW", "MID"], ["NEW", "OLD"]],
    }


def test_graph_view_drops_dangling_and_self_edges() -> None:
    doc = _doc()
    doc["edges"] += [["MID", "MISSING"], ["OLD", "OLD"]]
    view = graph_view(doc)
    assert len(view.nodes) == 4
    assert ("OLD", "OLD") not in view.edges
    assert all(u != v for u, v in view.edges)
    assert len(view.edges) == 3


def test_relevance_is_fraction_of_query_terms_in_title() -> None:
    view = graph_view(_doc())
    rel = relevance_vector(view, "surface code decoders")
    assert rel["NEW"] == 1.0  # all three terms present
    assert rel["OLD"] == 2 / 3  # "surface", "code"
    assert rel["DECOY"] == 0.0


def test_relevance_ignores_stopwords_only_query() -> None:
    view = graph_view(_doc())
    assert set(relevance_vector(view, "the and of").values()) == {0.0}


def test_personalised_pagerank_favours_the_seeded_region() -> None:
    view = graph_view(_doc())
    seeded = personalised_pagerank(view, {"OLD": 1.0, "MID": 0.0, "NEW": 0.0, "DECOY": 0.0})
    unseeded = personalised_pagerank(view, {"DECOY": 1.0})
    assert seeded["OLD"] > seeded["DECOY"]
    # The same node scores differently under a different seed — the whole point
    # of L5.2 (a query-blind ranking would return one fixed ordering).
    assert seeded["OLD"] != unseeded["OLD"]


def test_personalised_pagerank_falls_back_to_uniform_on_empty_seed() -> None:
    view = graph_view(_doc())
    scores = personalised_pagerank(view, {})
    assert sum(scores.values()) > 0
    assert len(scores) == 4


def test_highlight_selects_topical_nodes_and_their_edges() -> None:
    found = highlight(graph_view(_doc()), "surface code decoders", top_n=3)
    assert "DECOY" not in found.selected
    assert max(found.scores.values()) == 1.0  # rescaled to a 0..1 peak
    for u, v in found.edges:
        assert u in found.selected and v in found.selected


def test_anchor_is_earliest_among_relevant_not_earliest_overall() -> None:
    view = graph_view(_doc())
    rel = relevance_vector(view, "decoders")
    # DECOY (2001) is older than MID (2010) but irrelevant; MID must win.
    assert choose_anchor(view, rel, threshold=0.5) == "MID"


def test_anchor_threshold_gates_candidates() -> None:
    view = graph_view(_doc())
    rel = relevance_vector(view, "protein folding")
    assert choose_anchor(view, rel, threshold=0.5) == "DECOY"
    assert choose_anchor(view, rel, threshold=1.01) is None


def test_step_weights_blend_relevance_and_influence_by_alpha() -> None:
    view = graph_view(_doc())
    rel = relevance_vector(view, "surface code decoders")
    keyword_only = step_weights(view, rel, alpha=1.0)
    influence_only = step_weights(view, rel, alpha=0.0)
    assert keyword_only["NEW"] == 1.0
    assert influence_only["NEW"] == 0.3  # the node's influence, untouched
    assert influence_only["DECOY"] == 0.5


def test_broad_query_anchors_on_the_oldest_paper_it_matches() -> None:
    """L5.3.1 is "earliest *relevant*" — a broad query legitimately reaches furthest back.

    "surface code" matches OLD (1996) too, so OLD is the correct origin even though
    later papers match the keyword more completely. Narrowing the query is what
    moves the origin forward.
    """
    view = graph_view(_doc())
    paths = progression_paths(view, "surface code decoders", threshold=0.5)
    assert paths[0].anchor == "OLD"
    assert {step.leg for step in paths[0].steps} == {DEVELOPMENT}  # nothing older in-set


def test_progression_path_runs_both_ways_through_the_anchor() -> None:
    view = graph_view(_doc())
    paths = progression_paths(view, _MID_QUERY, alpha=0.5, threshold=_MID_THRESHOLD)
    assert paths
    best = paths[0]
    assert best.anchor == "MID"
    # Oldest first, anchor in the middle, later work after it.
    assert best.nodes.index("OLD") < best.nodes.index("MID") < best.nodes.index("NEW")
    legs = {step.leg for step in best.steps}
    assert legs == {ROOTS, DEVELOPMENT}


def test_roots_leg_survives_an_unmatching_foundational_paper() -> None:
    """OLD matches none of the keyword yet must still be reachable back through influence.

    This is why the floor applies to the blended weight, not raw relevance: a raw
    relevance floor would sever the roots leg the operator specifically asked for.
    """
    view = graph_view(_doc())
    assert relevance_vector(view, _MID_QUERY)["OLD"] == 0.0
    best = progression_paths(view, _MID_QUERY, alpha=0.5, threshold=_MID_THRESHOLD)[0]
    assert "OLD" in best.nodes


def test_path_edges_map_back_to_stored_orientation() -> None:
    view = graph_view(_doc())
    best = progression_paths(view, _MID_QUERY, threshold=_MID_THRESHOLD)[0]
    mapping = path_edges(best)
    # Stored edges are citing -> cited, both legs included.
    assert mapping[("MID", "OLD")] == ROOTS
    assert mapping[("NEW", "MID")] == DEVELOPMENT
    for edge in mapping:
        assert list(edge) in [list(e) for e in view.edges]


def test_no_anchor_yields_no_paths() -> None:
    view = graph_view(_doc())
    assert progression_paths(view, "topic absent from every title") == ()


def test_floor_stops_the_walk() -> None:
    view = graph_view(_doc())
    # A floor above every possible weight leaves the anchor stranded on its own.
    paths = progression_paths(view, _MID_QUERY, threshold=_MID_THRESHOLD, floor=1.5)
    assert paths
    assert paths[0].nodes == ("MID",)
    assert paths[0].steps == ()


def test_cycle_does_not_hang_the_walk() -> None:
    doc = _doc()
    doc["edges"] += [["OLD", "NEW"]]  # closes a cycle OLD -> NEW -> MID -> OLD
    paths = progression_paths(graph_view(doc), "surface code decoders", threshold=0.5)
    assert paths
    for path in paths:
        assert len(set(path.nodes)) == len(path.nodes)  # no node repeats


def test_connect_reports_a_directed_lineage() -> None:
    view = graph_view(_doc())
    report = connect(view, ["NEW", "OLD"])
    link = report.pairs[0]
    assert link.kind == LINEAGE
    assert link.path[0] == "NEW" and link.path[-1] == "OLD"
    assert report.all_connected is True
    # Overlay edges must exist in the stored graph or the renderer cannot draw them.
    for edge in report.edges:
        assert edge in view.edges


def test_connect_finds_the_shortest_chain() -> None:
    view = graph_view(_doc())
    # NEW cites OLD directly as well as via MID; the direct hop must win.
    assert connect(view, ["NEW", "OLD"]).pairs[0].path == ("NEW", "OLD")


def test_connect_reports_indirect_when_no_lineage_runs_either_way() -> None:
    doc = _doc()
    # Two papers that both cite OLD but never cite each other.
    doc["nodes"].append(_node("SIB", "Sibling work", 2011, 0.2))
    doc["edges"].append(["SIB", "OLD"])
    report = connect(graph_view(doc), ["SIB", "MID"])
    link = report.pairs[0]
    assert link.kind == INDIRECT
    assert "OLD" in link.path  # joined through their shared ancestor


def test_connect_reports_unconnected_across_components() -> None:
    view = graph_view(_doc())
    report = connect(view, ["DECOY", "MID"])  # DECOY has no edges at all
    assert report.pairs[0].kind == UNCONNECTED
    assert report.pairs[0].path == ()
    assert report.all_connected is False


def test_connect_handles_more_than_two_selections() -> None:
    report = connect(graph_view(_doc()), ["NEW", "MID", "OLD"])
    assert len(report.pairs) == 3  # every pair checked
    assert {"NEW", "MID", "OLD"} <= set(report.nodes)


def test_connect_deduplicates_and_ignores_a_single_selection() -> None:
    view = graph_view(_doc())
    assert connect(view, ["MID", "MID"]).pairs == ()
    assert connect(view, ["MID"]).all_connected is False


def _branching_doc() -> dict[str, Any]:
    """NEW reaches OLD three ways: direct, via MID, and via ALT."""
    doc = _doc()
    doc["nodes"].append(_node("ALT", "Alternate route", 2015, 0.4))
    doc["edges"] += [["NEW", "ALT"], ["ALT", "OLD"]]
    return doc


def test_expand_returns_every_lineage_shortest_first() -> None:
    view = graph_view(_branching_doc())
    link = connect(view, ["NEW", "OLD"], expand=True).pairs[0]
    assert len(link.lineages) == 3
    assert link.lineages[0] == ("NEW", "OLD")  # shortest first
    assert [len(route) for route in link.lineages] == sorted(
        len(route) for route in link.lineages
    )
    assert {"MID", "ALT"} <= set().union(*(set(r) for r in link.lineages))


def test_expand_highlights_every_route_not_only_the_shortest() -> None:
    view = graph_view(_branching_doc())
    plain = connect(view, ["NEW", "OLD"])
    expanded = connect(view, ["NEW", "OLD"], expand=True)
    assert len(expanded.pairs[0].edges) > len(plain.pairs[0].edges)
    assert set(expanded.nodes) >= {"NEW", "MID", "ALT", "OLD"}
    for edge in expanded.edges:
        assert edge in view.edges  # every overlay edge must be drawable


def test_expand_is_capped() -> None:
    view = graph_view(_branching_doc())
    link = connect(view, ["NEW", "OLD"], expand=True, max_paths=1).pairs[0]
    assert len(link.lineages) == 1


def test_expand_respects_the_depth_bound() -> None:
    view = graph_view(_branching_doc())
    # max_depth counts EDGES: 1 admits only the direct hop, 2 lets the detours in.
    assert connect(view, ["NEW", "OLD"], expand=True, max_depth=1).pairs[0].lineages == (
        ("NEW", "OLD"),
    )
    assert len(connect(view, ["NEW", "OLD"], expand=True, max_depth=2).pairs[0].lineages) == 3


def test_expand_terminates_on_a_cycle() -> None:
    doc = _branching_doc()
    doc["edges"].append(["OLD", "NEW"])  # cycle
    link = connect(graph_view(doc), ["NEW", "OLD"], expand=True).pairs[0]
    assert link.lineages
    for route in link.lineages:
        assert len(set(route)) == len(route)  # simple paths only


def test_expand_leaves_unconnected_pairs_alone() -> None:
    report = connect(graph_view(_doc()), ["DECOY", "MID"], expand=True)
    assert report.pairs[0].kind == UNCONNECTED
    assert report.pairs[0].lineages == ()


def test_alternatives_are_ranked_by_probability() -> None:
    doc = _doc()
    # Give MID a second, weaker line of development so a "Next" path exists.
    doc["nodes"].append(_node("ALT", "Surface code decoders alternative", 2020, 0.1))
    doc["edges"].append(["ALT", "MID"])
    paths = progression_paths(graph_view(doc), "surface code decoders", threshold=0.5, k=5)
    assert len(paths) > 1
    assert [p.score for p in paths] == sorted((p.score for p in paths), reverse=True)
