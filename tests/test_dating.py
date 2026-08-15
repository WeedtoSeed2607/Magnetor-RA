"""Dates carry how precise they are, and refuse questions they cannot answer."""

from __future__ import annotations

import datetime as dt

from magnetor.dating import (
    DateQuality,
    Placement,
    place,
    placeable,
    resolves,
    span_days,
    tally,
)
from magnetor.types import DatePrecision

_NOW = dt.datetime(2026, 8, 13, tzinfo=dt.UTC)


def _place(
    published: dt.datetime | None,
    precision: DatePrecision | None,
    *,
    window_days: int = 30,
) -> Placement:
    return place(published, precision, now=_NOW, window_days=window_days)


def test_a_day_exact_date_can_answer_a_thirty_day_question() -> None:
    assert resolves(DatePrecision.DAY, 30)
    assert _place(_NOW - dt.timedelta(days=5), DatePrecision.DAY) is Placement.RECENT
    assert _place(_NOW - dt.timedelta(days=90), DatePrecision.DAY) is Placement.EARLIER


def test_a_year_only_date_cannot_be_placed_in_a_thirty_day_window() -> None:
    """"Sometime in 2026" sits on both sides of any cutoff inside 2026."""
    assert not resolves(DatePrecision.YEAR, 30)
    assert _place(dt.datetime(2026, 1, 1, tzinfo=dt.UTC), DatePrecision.YEAR) is (
        Placement.TOO_COARSE
    )


def test_a_month_date_answers_a_wide_window_but_not_a_narrow_one() -> None:
    stamp = _NOW - dt.timedelta(days=5)
    assert _place(stamp, DatePrecision.MONTH, window_days=30) is Placement.TOO_COARSE
    assert _place(stamp, DatePrecision.MONTH, window_days=90) is Placement.RECENT


def test_precision_spans_are_ordered_widest_last() -> None:
    assert (
        span_days(DatePrecision.DAY)
        < span_days(DatePrecision.MONTH)
        < span_days(DatePrecision.YEAR)
    )


def test_a_paper_dated_in_the_future_is_not_part_of_what_is_published_now() -> None:
    """PubMed's pubdate is the issue date, so ahead-of-print lands after today."""
    ahead = _NOW + dt.timedelta(days=140)
    assert _place(ahead, DatePrecision.MONTH) is Placement.NOT_YET_PUBLISHED
    assert _place(ahead, DatePrecision.DAY) is Placement.NOT_YET_PUBLISHED


def test_a_missing_date_is_reported_not_guessed() -> None:
    assert _place(None, None) is Placement.UNDATED
    assert _place(None, DatePrecision.DAY) is Placement.UNDATED


def test_unrecorded_precision_is_placed_rather_than_rejected() -> None:
    """Legacy records predate the field; refusing them would blank a corpus."""
    assert _place(_NOW - dt.timedelta(days=5), None) is Placement.RECENT
    assert _place(_NOW - dt.timedelta(days=90), None) is Placement.EARLIER


def test_placeable_agrees_with_place() -> None:
    recent = _NOW - dt.timedelta(days=5)
    assert placeable(recent, DatePrecision.DAY, now=_NOW, window_days=30)
    assert not placeable(recent, DatePrecision.YEAR, now=_NOW, window_days=30)
    assert not placeable(None, None, now=_NOW, window_days=30)
    assert not placeable(
        _NOW + dt.timedelta(days=1), DatePrecision.DAY, now=_NOW, window_days=30
    )


def test_tally_separates_what_was_used_from_what_was_dropped() -> None:
    quality = tally(
        {
            "a": Placement.RECENT,
            "b": Placement.RECENT,
            "c": Placement.EARLIER,
            "d": Placement.UNDATED,
            "e": Placement.TOO_COARSE,
            "f": Placement.NOT_YET_PUBLISHED,
        },
        unrecorded=2,
    )
    assert (quality.recent, quality.earlier) == (2, 1)
    assert quality.placed == 3
    assert quality.excluded == 3
    assert quality.total == 6
    assert quality.precision_unrecorded == 2


def test_the_note_names_every_reason_a_paper_was_dropped() -> None:
    note = DateQuality(
        recent=10, earlier=5, undated=1, too_coarse=2, not_yet_published=3
    ).note()
    assert "1 have no publication date" in note
    assert "2 are dated by year or month only" in note
    assert "3 are dated in the future" in note
    assert "6 of 21" in note


def test_no_note_when_every_paper_was_usable() -> None:
    assert DateQuality(recent=10, earlier=5).note() == ""
