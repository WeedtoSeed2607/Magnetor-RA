"""Branch C — breadth assessment and narrowing suggestions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from magnetor.narrowing import BROAD, FOCUSED, SCATTERED, Suggestion, assess


def _graph(titles: dict[str, str], edges: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "query": "test",
        "nodes": [
            {"id": nid, "title": title, "year": 2000, "influence": 0.5}
            for nid, title in titles.items()
        ],
        "edges": [[u, v] for u, v in edges],
    }


def _clique(prefix: str, size: int, title: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    ids = [f"{prefix}{i}" for i in range(size)]
    return ({i: title for i in ids}, list(combinations(ids, 2)))


def test_a_dense_single_component_reads_as_focused() -> None:
    titles, edges = _clique("A", 8, "surface code decoders")
    report = assess(_graph(titles, edges))
    assert report.verdict == FOCUSED
    assert report.components == 1
    assert report.pair_coverage == 1.0


def test_two_disconnected_literatures_read_as_scattered() -> None:
    left, left_edges = _clique("L", 6, "surface code decoders")
    right, right_edges = _clique("R", 6, "protein folding kinetics")
    report = assess(_graph({**left, **right}, left_edges + right_edges))
    assert report.verdict == SCATTERED
    assert report.largest_component_share == 0.5


def test_a_single_straggler_does_not_flip_the_verdict() -> None:
    """Fragmentation is a share, not a component count.

    A 21-node graph with one detached node has two components and is plainly one
    literature; counting components would call it broad.
    """
    titles, edges = _clique("A", 20, "surface code decoders")
    titles["LONE"] = "unrelated work"
    report = assess(_graph(titles, edges))
    assert report.components == 2
    assert report.verdict == FOCUSED


def test_a_sparse_graph_reads_as_broad() -> None:
    titles = {f"N{i}": "assorted topic" for i in range(20)}
    edges = [(f"N{i}", f"N{i + 1}") for i in range(0, 18, 2)]  # a few disjoint pairs
    report = assess(_graph(titles, edges))
    assert report.verdict in (BROAD, SCATTERED)
    assert report.pair_coverage < 0.08


def test_suggestions_pick_the_cohesive_subtopic() -> None:
    """The sub-literature that cites internally wins over the merely frequent word."""
    titles, edges = _clique("D", 6, "quantum decoder threshold")
    # Ten more papers share a common word but cite nobody.
    titles.update({f"X{i}": "quantum assorted miscellany" for i in range(10)})
    report = assess(_graph(titles, edges), min_cohesion=1.0)
    phrases = [s.phrase for s in report.suggestions]
    assert any("decoder" in p for p in phrases)
    # "quantum" spans everything, so it cannot narrow anything.
    assert "quantum" not in phrases


def test_a_phrase_covering_almost_everything_is_rejected() -> None:
    titles, edges = _clique("A", 8, "surface code decoders")
    report = assess(_graph(titles, edges), max_share=0.5)
    assert all(s.share <= 0.5 for s in report.suggestions)


def test_nested_phrases_collapse_to_the_specific_one() -> None:
    titles, edges = _clique("A", 6, "medical image segmentation")
    titles.update({f"B{i}": "unrelated topic here" for i in range(10)})
    report = assess(_graph(titles, edges), min_cohesion=1.0)
    phrases = {s.phrase for s in report.suggestions}
    # "medical", "medical image" and "medical image segmentation" cover the same
    # papers; only one survives.
    assert sum(1 for p in phrases if "medical" in p) == 1


def test_score_damps_small_groups_and_prefers_phrases() -> None:
    tiny = Suggestion(phrase="dog", papers=3, share=0.05, cohesion=6.0, examples=())
    solid = Suggestion(phrase="moral judgment", papers=12, share=0.2, cohesion=3.0, examples=())
    assert solid.score > tiny.score  # shrinkage beats a spectacular small-sample ratio
    one_word = Suggestion(phrase="vision", papers=6, share=0.1, cohesion=3.0, examples=())
    two_words = Suggestion(phrase="computer vision", papers=6, share=0.1, cohesion=3.0, examples=())
    assert two_words.score > one_word.score


def test_thresholds_are_parameters_not_baked_in() -> None:
    titles, edges = _clique("A", 8, "surface code decoders")
    graph = _graph(titles, edges)
    assert assess(graph).verdict == FOCUSED
    # Demanding near-total coverage reclassifies the same graph, which is the
    # point: the cutoff is a declared convention, not a measurement.
    assert assess(graph, broad_coverage=1.5).verdict == BROAD


def test_empty_and_tiny_graphs_are_handled() -> None:
    empty = assess({"nodes": [], "edges": []})
    assert empty.verdict == FOCUSED
    assert empty.suggestions == ()
    single = assess(_graph({"A": "one paper"}, []))
    assert single.suggestions == ()
