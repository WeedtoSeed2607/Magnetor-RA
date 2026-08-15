"""Branch A — grouping papers by meaning rather than by shared wording."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np

from magnetor.themes import discover_themes
from magnetor.types import DatePrecision, Domain, Paper

_NOW = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)


def _paper(
    pid: str,
    title: str,
    abstract: str,
    *,
    days_ago: int,
    precision: DatePrecision | None = DatePrecision.DAY,
) -> Paper:
    return Paper(
        domain=Domain.QUANTUM_MECHANICS,
        source="test",
        external_id=pid,
        title=title,
        abstract=abstract,
        authors=("A",),
        published=_NOW - dt.timedelta(days=days_ago),
        published_precision=precision,
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


def test_latex_markup_never_reaches_a_label() -> None:
    r"""``\mathbb{R}`` tokenised to *mathbb*, which labelled the largest maths group."""
    ids, vectors, papers = _corpus()
    for pid in ids[:9]:
        papers[pid] = dataclasses.replace(
            papers[pid],
            abstract=r"We bound $\mathbb{R}^n$ using $\mathcal{O}(\log n)$ steps.",
        )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    words = {w for theme in report.themes for w in theme.label}
    assert not {"mathbb", "mathcal"} & words


def test_an_ordinary_word_is_kept_when_it_is_not_a_latex_command() -> None:
    """Stripping the command form must not blind the labels to the plain word."""
    ids, vectors, papers = _corpus()
    for pid in ids[:9]:
        papers[pid] = dataclasses.replace(
            papers[pid], abstract="The log spiral describes log growth in log time."
        )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert "log" in {w for theme in report.themes for w in theme.label}


def test_function_words_never_label_a_group() -> None:
    """Words that cluster with prose style, not subject — the one case the
    outside-rate subtraction cannot catch."""
    ids, vectors, papers = _corpus()
    for pid in ids[:9]:
        papers[pid] = dataclasses.replace(
            papers[pid],
            abstract="There exists a bound for every case, but only when it holds.",
        )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    words = {w for theme in report.themes for w in theme.label}
    assert not {"there", "but", "every", "only"} & words


def test_a_year_only_date_is_still_grouped_but_left_out_of_the_comparison() -> None:
    """The paper belongs to its theme; it just cannot say which side it fell on."""
    ids, vectors, papers = _corpus()
    for pid in ids[:4]:
        papers[pid] = dataclasses.replace(
            papers[pid], published_precision=DatePrecision.YEAR
        )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.dates.too_coarse == 4
    # Grouped: every paper still counts towards a theme's size...
    assert sum(t.size for t in report.themes) == len(ids)
    # ...but only the placeable ones carry a before/after claim.
    assert sum(t.recent + t.earlier for t in report.themes) == len(ids) - 4


def test_a_paper_dated_in_the_future_is_not_counted_as_recent() -> None:
    """PubMed issue dates run ahead of print; that is not published *now*."""
    ids, vectors, papers = _corpus()
    papers[ids[0]] = dataclasses.replace(
        papers[ids[0]],
        published=_NOW + dt.timedelta(days=140),
        published_precision=DatePrecision.MONTH,
    )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.dates.not_yet_published == 1
    assert report.dates.placed == len(ids) - 1


def test_shares_still_sum_to_one_when_papers_were_excluded() -> None:
    """Shares are of what could be placed, so an exclusion must not unbalance them."""
    ids, vectors, papers = _corpus()
    for pid in ids[:3]:
        papers[pid] = dataclasses.replace(
            papers[pid], published=None, published_precision=None
        )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.dates.undated == 3
    assert sum(t.share_now for t in report.themes) == 1.0
    assert sum(t.share_before for t in report.themes) == 1.0


def test_a_corpus_shorter_than_the_window_is_reported_as_such() -> None:
    """One week of arXiv cannot support a 30-day before-and-after claim."""
    ids, vectors, papers = _corpus()
    for i, pid in enumerate(ids):
        papers[pid] = dataclasses.replace(
            papers[pid], published=_NOW - dt.timedelta(days=30 + (i % 7))
        )
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.span_days < report.window_days
    assert not report.spans_the_window


def test_groups_are_ordered_by_size_when_growth_cannot_be_claimed() -> None:
    """With no earlier stretch to compare against, "growing first" is not a fact.

    Ordering is a claim too: putting a group at the top says it is rising. When
    every paper arrived in one batch, size is the only true thing left to sort on.
    """
    ids: list[str] = []
    vectors: list[list[float]] = []
    papers: dict[str, Paper] = {}
    # A small group and a large one, every paper inside a single week.
    for prefix, count, vector, words in (
        ("S", 6, [1.0, 0.0], ("Decoherence in superconducting qubits", "Qubit decay.")),
        ("B", 12, [0.0, 1.0], ("Homology of algebraic manifolds", "Manifold homology.")),
    ):
        for i in range(count):
            pid = f"{prefix}{i}"
            ids.append(pid)
            vectors.append(vector)
            papers[pid] = _paper(pid, words[0], words[1], days_ago=31 + (i % 7))

    report = discover_themes(
        ids, np.asarray(vectors, dtype=np.float32), papers, n_themes=2, now=_NOW
    )
    assert not report.spans_the_window
    assert [t.size for t in report.themes] == [12, 6]


def test_a_corpus_spanning_the_window_is_reported_as_such() -> None:
    ids, vectors, papers = _corpus()
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.span_days >= report.window_days
    assert report.spans_the_window


def test_records_without_a_recorded_precision_are_counted_not_assumed_exact() -> None:
    """Legacy records are usable, but the uncertainty stays visible."""
    ids, vectors, papers = _corpus()
    for pid in ids:
        papers[pid] = dataclasses.replace(papers[pid], published_precision=None)
    report = discover_themes(ids, vectors, papers, n_themes=2, now=_NOW)
    assert report.dates.precision_unrecorded == len(ids)
    assert report.dates.placed == len(ids)


def test_grouping_is_deterministic() -> None:
    """Same corpus, same groups — otherwise no claim about growth is reproducible."""
    ids, vectors, papers = _corpus()
    first = discover_themes(ids, vectors, papers, n_themes=2)
    second = discover_themes(ids, vectors, papers, n_themes=2)
    assert [t.examples for t in first.themes] == [t.examples for t in second.themes]
    assert [t.label for t in first.themes] == [t.label for t in second.themes]
