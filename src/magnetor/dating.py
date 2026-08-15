"""How precisely a paper was dated, and what that lets a caller claim.

Sources do not all know when a paper appeared. PubMed's ``pubdate`` may be
``2026 Mar 04``, ``2026 Mar`` or just ``2026``; arXiv always gives a timestamp.
Parsing hides the difference, because :func:`datetime.strptime` fills whatever
is missing with January 1st — so *"sometime in 2026"* and *"January 1st, 2026"*
become the same value and nothing downstream can tell them apart.

That is the failure this module exists to prevent. It is C3 (absence is
displayed, not imputed) applied to time: a date carries how precise it is, and a
caller asking a question finer than the date can answer gets
:attr:`Placement.TOO_COARSE` rather than a confident wrong answer.

**Two separate defects live here, not one.**

*Imprecision.* A year-only date cannot be placed inside a 30-day window — the
paper could fall on either side of the cutoff and the record does not say. The
honest answer is to leave it out of the comparison and count it, which is what
:class:`DateQuality` is for.

*Future dates.* PubMed's ``pubdate`` is the **issue** date, so an ahead-of-print
article can be dated months from now. :mod:`magnetor.pipeline` already caps the
harvest watermark against this, or a single such paper would push the ``since``
floor past today and quietly stop the domain from ever fetching again. That
defence protects acquisition only; analysis needs its own, because a paper that
has not been published yet is not part of what is being published *now*.

**Records written before precision was tracked** carry ``None``, which means *not
recorded* — not *day-exact*. They are still placed, because refusing to would
blank an existing corpus, but they are counted separately so the uncertainty is
visible rather than assumed away. Re-harvesting a domain clears the count.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

from magnetor.types import DatePrecision

#: How much time a date of each precision could actually refer to. A month-precise
#: date names up to 31 days; a year-precise one up to 366. These are the widths a
#: window has to exceed before the date can be placed inside it.
_SPAN_DAYS: dict[DatePrecision, int] = {
    DatePrecision.DAY: 1,
    DatePrecision.MONTH: 31,
    DatePrecision.YEAR: 366,
}


class Placement(StrEnum):
    """Where one paper falls relative to a window — or why it cannot be said."""

    RECENT = "recent"
    EARLIER = "earlier"
    #: No publication date at all.
    UNDATED = "undated"
    #: Dated, but too coarsely to sit on one side of the cutoff.
    TOO_COARSE = "too_coarse"
    #: Dated after the moment of asking. Ahead-of-print issue dates do this.
    NOT_YET_PUBLISHED = "not_yet_published"


def span_days(precision: DatePrecision) -> int:
    """Widest stretch of time a date of this precision could mean."""
    return _SPAN_DAYS[precision]


def resolves(precision: DatePrecision, window_days: int) -> bool:
    """Whether this precision can place a paper inside a window that long.

    A month-precise date cannot answer "in the last 30 days": *2026 Jul* spans
    both sides of any cutoff falling inside July, and the record does not say
    which. Comparing spans is the whole test.
    """
    return span_days(precision) <= window_days


def place(
    published: dt.datetime | None,
    precision: DatePrecision | None,
    *,
    now: dt.datetime,
    window_days: int,
) -> Placement:
    """Decide whether one paper counts as recent, earlier, or neither.

    ``precision=None`` means a record predating precision tracking. It is placed
    rather than rejected — see the module docstring — so only an explicitly
    recorded coarse precision can produce :attr:`Placement.TOO_COARSE`.
    """
    if published is None:
        return Placement.UNDATED
    if published > now:
        return Placement.NOT_YET_PUBLISHED
    if precision is not None and not resolves(precision, window_days):
        return Placement.TOO_COARSE
    cutoff = now - dt.timedelta(days=window_days)
    return Placement.RECENT if published >= cutoff else Placement.EARLIER


def placeable(
    published: dt.datetime | None,
    precision: DatePrecision | None,
    *,
    now: dt.datetime,
    window_days: int,
) -> bool:
    """Whether a date is present, past and precise enough to be bucketed.

    The same test the time-slicing in :mod:`magnetor.trends` needs: a paper that
    cannot be put on one side of a cutoff ``window_days`` wide equally cannot be
    assigned to a slice that wide.
    """
    return place(published, precision, now=now, window_days=window_days) in (
        Placement.RECENT,
        Placement.EARLIER,
    )


@dataclass(frozen=True, slots=True)
class DateQuality:
    """What a windowed comparison could and could not use, and why.

    Carried alongside any before/after figure so a reader can see what the
    figure was computed on. A percentage over 12 of 130 papers is a different
    claim from the same percentage over all 130.
    """

    recent: int = 0
    earlier: int = 0
    undated: int = 0
    too_coarse: int = 0
    not_yet_published: int = 0
    #: Placed, but from records written before precision was tracked.
    precision_unrecorded: int = 0

    @property
    def placed(self) -> int:
        """Papers that could be put on one side of the cutoff."""
        return self.recent + self.earlier

    @property
    def excluded(self) -> int:
        return self.undated + self.too_coarse + self.not_yet_published

    @property
    def total(self) -> int:
        return self.placed + self.excluded

    def note(self) -> str:
        """One plain sentence about what was left out. Empty when nothing was."""
        if not self.excluded:
            return ""
        reasons = []
        if self.undated:
            reasons.append(f"{self.undated} have no publication date")
        if self.too_coarse:
            reasons.append(
                f"{self.too_coarse} are dated by year or month only, so they "
                "cannot be placed on either side of the cutoff"
            )
        if self.not_yet_published:
            reasons.append(
                f"{self.not_yet_published} are dated in the future, which is the "
                "journal issue date rather than the day the paper appeared"
            )
        return (
            f"{self.excluded} of {self.total} papers are left out of the "
            f"before-and-after comparison: " + "; ".join(reasons) + "."
        )


def tally(placements: dict[str, Placement], unrecorded: int = 0) -> DateQuality:
    """Count placements into a :class:`DateQuality`."""
    counts = {outcome: 0 for outcome in Placement}
    for placement in placements.values():
        counts[placement] += 1
    return DateQuality(
        recent=counts[Placement.RECENT],
        earlier=counts[Placement.EARLIER],
        undated=counts[Placement.UNDATED],
        too_coarse=counts[Placement.TOO_COARSE],
        not_yet_published=counts[Placement.NOT_YET_PUBLISHED],
        precision_unrecorded=unrecorded,
    )
