"""When a map's own links contradict its dates.

An edge points citing → cited, so the cited paper cannot carry a later year than
the paper citing it. Across the nine graphs stored on this machine, 129 of 1,883
datable edges break that rule — 6.85% overall, and 31% in the worst single graph.
A reader following ``roots ← anchor → development`` is reading a claim about
time, so that rate is something they need told.

**Three things produce it, and none is a fault in this code.** A gap of decades
is a reissue: Mead's *Mind, Self, and Society* (1934) is recorded here as 2015
and Piaget's *The Moral Judgment Of The Child* (1932) as 2013, each taking the
year of the edition OpenAlex indexed. A gap of a year or two is usually version
lag — the citing author read a preprint and the year here is the journal
version's. And some edges are simply wrong: a 2018 paper in the transformer graph
cites a 2023 one, which no dating convention explains.

Because the causes differ, the note deliberately does not name one.

**Why this reports rather than repairs.** The obvious repair is to lower a
contradicted year to the earliest year among the papers citing it, which the
citation graph proves is a genuine upper bound. Measured against the nine stored
graphs, that lowers 24 node-years and changes the earliest paper in **none** of
them: the inflated dates are still later than the genuinely oldest paper in each
map. So the repair invents a number no source stated and buys no better answer.
What a reader actually needs is to know the dates are soft — which is C3 again,
unreliability displayed rather than imputed away.

Pure, in the style of :mod:`magnetor.diagnostics`: the caller gathers the years
and edges, so every branch is reachable in a test without a stored graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Impossible:
    """One edge whose two years cannot both be right."""

    citing: str
    citing_year: int
    cited: str
    cited_year: int

    @property
    def gap(self) -> int:
        """Years by which the cited paper post-dates the paper citing it."""
        return self.cited_year - self.citing_year


@dataclass(frozen=True, slots=True)
class ChronologyReport:
    """What the citation links say about how much the dates can be trusted."""

    impossible: tuple[Impossible, ...]
    #: Edges where both ends carried a year, so the check could be made at all.
    checked: int
    #: Edges skipped because an end had no year — not evidence of soundness.
    undatable: int

    @property
    def rate(self) -> float:
        return len(self.impossible) / self.checked if self.checked else 0.0

    @property
    def sound(self) -> bool:
        """True only when edges were checked and none contradicted its dates."""
        return self.checked > 0 and not self.impossible

    @property
    def widest(self) -> Impossible | None:
        """The starkest contradiction, which is the one worth showing a reader."""
        return max(self.impossible, key=lambda i: i.gap, default=None)

    def note(self) -> str:
        """One plain paragraph for a reader. Empty when nothing contradicts."""
        if not self.impossible:
            return ""
        return (
            f"{len(self.impossible)} of the {self.checked} links in this map run "
            f"the wrong way in time: they show a paper citing work dated after it, "
            f"which cannot happen. Usually the later date is not when the work "
            f"first appeared — it can be a reprint, a journal version of something "
            f"first shared as a preprint, or a mistake in the source record. Read "
            f"the dates here as approximate."
        )


def check(
    years: Mapping[str, int | None], edges: Sequence[tuple[str, str]]
) -> ChronologyReport:
    """Find every edge whose two years cannot both be right.

    ``edges`` are stored citing → cited. An edge with a missing year at either
    end is counted as undatable rather than sound: not looking is not the same
    as looking and finding nothing wrong.
    """
    impossible: list[Impossible] = []
    checked = undatable = 0
    for citing, cited in edges:
        citing_year, cited_year = years.get(citing), years.get(cited)
        if citing_year is None or cited_year is None:
            undatable += 1
            continue
        checked += 1
        if cited_year > citing_year:
            impossible.append(
                Impossible(
                    citing=citing,
                    citing_year=citing_year,
                    cited=cited,
                    cited_year=cited_year,
                )
            )
    impossible.sort(key=lambda i: (-i.gap, i.citing, i.cited))
    return ChronologyReport(tuple(impossible), checked=checked, undatable=undatable)
