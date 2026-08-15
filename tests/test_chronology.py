"""A map's citation links are evidence about how far its dates can be trusted."""

from __future__ import annotations

from magnetor.chronology import check


def test_an_edge_pointing_forward_in_time_is_impossible() -> None:
    """A paper cannot cite work published after it."""
    report = check({"new": 1992, "old": 2009}, [("new", "old")])
    assert not report.sound
    assert len(report.impossible) == 1
    assert report.impossible[0].gap == 17


def test_an_ordinary_citation_raises_nothing() -> None:
    report = check({"new": 2009, "old": 1992}, [("new", "old")])
    assert report.sound
    assert report.impossible == ()
    assert report.rate == 0.0


def test_same_year_citations_are_allowed() -> None:
    """Papers do cite work from their own year; only *later* is impossible."""
    report = check({"a": 2009, "b": 2009}, [("a", "b")])
    assert report.sound


def test_an_edge_with_a_missing_year_is_undatable_not_sound() -> None:
    """Not looking is not the same as looking and finding nothing wrong."""
    report = check({"a": 2009, "b": None}, [("a", "b")])
    assert report.checked == 0
    assert report.undatable == 1
    assert not report.sound  # nothing was verified, so nothing is claimed
    assert report.note() == ""


def test_an_unknown_node_is_treated_as_undatable() -> None:
    report = check({"a": 2009}, [("a", "missing")])
    assert report.undatable == 1
    assert report.checked == 0


def test_rate_is_measured_against_edges_actually_checked() -> None:
    years = {"a": 2000, "b": 2010, "c": 1990}
    report = check(years, [("a", "b"), ("a", "c")])
    assert report.checked == 2
    assert report.rate == 0.5


def test_the_widest_contradiction_is_the_one_surfaced() -> None:
    years = {"a": 1992, "b": 2009, "c": 1993, "d": 1994}
    report = check(years, [("c", "d"), ("a", "b")])
    widest = report.widest
    assert widest is not None
    assert (widest.citing, widest.cited) == ("a", "b")
    assert report.impossible[0] is widest  # sorted worst-first


def test_no_contradictions_means_no_note() -> None:
    assert check({"a": 2009, "b": 1992}, [("a", "b")]).note() == ""


def test_the_note_gives_the_count_the_total_and_the_possible_reasons() -> None:
    """Three causes produce these, so the note must not settle on one."""
    years = {"a": 1992, "b": 2009, "c": 2010, "d": 1990}
    note = check(years, [("a", "b"), ("c", "d")]).note()
    assert "1 of the 2 links" in note
    assert "reprint" in note
    assert "preprint" in note
    assert "mistake in the source record" in note
    assert "approximate" in note
