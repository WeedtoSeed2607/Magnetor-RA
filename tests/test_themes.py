"""Branch A — grouping papers by meaning rather than by shared wording."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np

from magnetor.themes import discover_themes
from magnetor.types import Domain, Paper

_NOW = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)


def _paper(pid: str, title: str, abstract: str, *, days_ago: int) -> Paper:
    return Paper(
        domain=Domain.QUANTUM_MECHANICS,
        source="test",
        external_id=pid,
        title=title,
        abstract=abstract,
        authors=("A",),
        published=_NOW - dt.timedelta(days=days_ago),
        doi=None,
    )


def _corpus(recent_left: int = 6, recent_right: int = 6):
    """Two clearly separated groups, sitting at opposite ends of a 2-D space.

    Vectors are unit length, as the real index stores them.
    """
    ids: list[str] = []
    vectors: list[list[float]] = []
    papers: dict[str, Paper] = {}
    for i in range(9):
        pid = f"L{i}"
        ids.append(pid)
        vectors.append([1.0, 0.0])
        papers[pid] = _paper(
            pid, "Decoherence in superconducting qubits",
            "Superconducting qubit decoherence measured.",
            days_ago=5 if i < recent_left else 200,
        )
    for i in range(9):
        pid = f"R{i}"
        ids.append(pid)
        vectors.append([0.0, 1.0])
        papers[pid] = _paper(
            pid, "Homology of algebraic manifolds",
            "Algebraic manifold homology computed.",
            days_ago=5 if i < recent_right else 200,
        )
    return ids, np.asarray(vectors, dtype=np.float32), papers


def test_papers_are_grouped_by_meaning() -> None:
    ids, vectors, papers = _corpus()
    report = discover_themes(ids, vectors, papers, n_themes=2)
    assert report.usable
    assert len(report.themes) == 2
    # Each group holds one side and nothing from the other.
    for theme in report.themes:
        prefixes = {pid[0] for pid in theme.examples}
        assert len(prefixes) == 1


def test_labels_drop_words_shared_across_groups() -> None:
    """A word every group uses cannot distinguish one, so it must not label one."""
    ids, vectors, papers = _corpus()
    # Give both sides a word in common.
    for pid, paper in papers.items():
        papers[pid] = dataclasses.replace(paper, abstract=f"{paper.abstract} lattice")
    report = discover_themes(ids, vectors, papers, n_themes=2)
    for theme in report.themes:
        assert "lattice" not in theme.label


def test_labels_use_words_distinctive_to_the_group() -> None:
    ids, vectors, papers = _corpus()
    report = discover_themes(ids, vectors, papers, n_themes=2)
    words = {w for theme in report.themes for w in theme.label}
    assert {"decoherence", "superconducting"} & words
    assert {"homology", "algebraic", "manifold", "manifolds"} & words


def test_a_group_growing_recently_is_reported_first() -> None:
    """The question is what is happening now, not what is biggest overall."""
    ids, vectors, papers = _corpus(recent_left=8, recent_right=1)
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.themes[0].drift > report.themes[1].drift
    assert report.themes[0].examples[0].startswith("L")


def test_shares_are_measured_against_each_period_not_the_whole(  ) -> None:
    ids, vectors, papers = _corpus(recent_left=8, recent_right=1)
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert sum(t.share_now for t in report.themes) == 1.0
    assert sum(t.share_before for t in report.themes) == 1.0


def test_examples_are_the_papers_nearest_the_group_centre() -> None:
    ids, vectors, papers = _corpus()
    report = discover_themes(ids, vectors, papers, n_themes=2, examples=2)
    for theme in report.themes:
        assert len(theme.examples) == 2
        assert set(theme.examples) <= set(ids)


def test_too_few_papers_reports_why_rather_than_inventing_groups() -> None:
    ids, vectors, papers = _corpus()
    report = discover_themes(ids[:4], vectors[:4], papers, n_themes=6)
    assert not report.usable
    assert "too few" in report.reason
    assert report.themes == ()


def test_ids_without_a_matching_record_are_ignored() -> None:
    ids, vectors, papers = _corpus()
    del papers[ids[0]]
    report = discover_themes(ids, vectors, papers, n_themes=2)
    assert report.papers == len(ids) - 1


def test_grouping_is_deterministic() -> None:
    """Same corpus, same groups — otherwise no claim about growth is reproducible."""
    ids, vectors, papers = _corpus()
    first = discover_themes(ids, vectors, papers, n_themes=2)
    second = discover_themes(ids, vectors, papers, n_themes=2)
    assert [t.examples for t in first.themes] == [t.examples for t in second.themes]
    assert [t.label for t in first.themes] == [t.label for t in second.themes]
