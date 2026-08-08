"""Branch C — open gaps: what a map is missing, and which approaches are unexplored.

Two questions the Evidence Graph can answer about its own incompleteness, both
computable from data the harvest already holds.

**Foundational gaps — the works you lean on but never captured.** Boundary
leakage says *how much* of the citation record points outside the harvested set;
it does not say *what*. Counting how many in-set papers cite each outside work
turns that number into a list: a work cited by seven of your papers and absent
from your map is not noise, it is the thing your map is missing. STATUS records
this under a name worth keeping — "foundational completeness" — and notes it is
the honest replacement for raw leakage, because leakage counts a long tail of
singleton references that nobody would call a gap.

**Facet gaps — the axes nobody has taken.** If a question has been attacked
twelve times empirically and once formally, the formal axis is not settled, it is
unexplored. That is the operator's "degree of freedom, axis-wise" made countable:
not a claim that the missing work would succeed, only that the citation record
shows nobody has tried it here.

**What this is not.** Neither measure estimates whether filling a gap would be
*fruitful*. A gap is an absence in a harvested slice, and absence has cheap
explanations — the work exists but was not indexed, or the approach is
inapplicable, or it was tried and failed to be cited. Read this as a prompt for
attention, never as a verdict about the literature.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from magnetor.facets import FACET_LABELS, UNCLASSIFIED, node_facets
from magnetor.harvest import HarvestResult
from magnetor.relations import reference_map


class WorkFetcher(Protocol):
    """The one capability gap enrichment needs.

    Narrower than either ``WorksSource`` or ``NeighbourSource`` on purpose: both
    satisfy it, so keyword and anchored harvests share this path without either
    protocol having to know about the other.
    """

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        ...

#: In-set papers that must cite an outside work before it counts as a gap rather
#: than a passing reference. Two is the floor the STATUS entry names; it is a
#: declared convention, not a calibrated threshold.
DEFAULT_MIN_CITATIONS = 2
DEFAULT_GAP_LIMIT = 25

#: A facet held by at most this share of papers reads as thin rather than covered.
DEFAULT_THIN_SHARE = 0.10


@dataclass(frozen=True, slots=True)
class FoundationalGap:
    """An outside work that the harvested set repeatedly leans on."""

    openalex_id: str
    cited_by_in_set: int
    share: float  # fraction of harvested papers citing it
    title: str | None = None
    year: int | None = None

    @property
    def url(self) -> str:
        return f"https://openalex.org/{self.openalex_id}"


@dataclass(frozen=True, slots=True)
class FacetGap:
    facet: str
    papers: int
    share: float
    absent: bool


def foundational_gaps(
    result: HarvestResult,
    *,
    min_citations: int = DEFAULT_MIN_CITATIONS,
    limit: int = DEFAULT_GAP_LIMIT,
) -> tuple[FoundationalGap, ...]:
    """Outside works cited by ``min_citations`` or more of the harvested papers.

    Identifiers and counts only. Titles need a separate fetch, since by
    definition these works were never retrieved — see :func:`enrich`.
    """
    references = reference_map(result)
    in_set = set(references)
    total = len(references)
    if not total:
        return ()

    demand: Counter[str] = Counter()
    for refs in references.values():
        for ref in refs:
            if ref not in in_set:
                demand[ref] += 1

    found = [
        FoundationalGap(
            openalex_id=work_id, cited_by_in_set=count, share=count / total
        )
        for work_id, count in demand.items()
        if count >= min_citations
    ]
    found.sort(key=lambda gap: (-gap.cited_by_in_set, gap.openalex_id))
    return tuple(found[:limit])


def enrich(
    gaps: Sequence[FoundationalGap], works: Sequence[Mapping[str, object]]
) -> tuple[FoundationalGap, ...]:
    """Attach titles and years from fetched works, leaving unknowns as ``None``.

    A gap whose metadata could not be retrieved keeps its identifier and count:
    dropping it would quietly shrink the very measure of incompleteness this
    exists to report.
    """
    by_id: dict[str, Mapping[str, object]] = {}
    for work in works:
        raw = str(work.get("id") or "")
        if raw:
            by_id[raw.rsplit("/", 1)[-1]] = work
    out = []
    for gap in gaps:
        found = by_id.get(gap.openalex_id)
        title = None
        year = None
        if found is not None:
            title = str(found.get("display_name") or found.get("title") or "") or None
            raw_year = found.get("publication_year")
            year = raw_year if isinstance(raw_year, int) else None
        out.append(
            FoundationalGap(
                openalex_id=gap.openalex_id,
                cited_by_in_set=gap.cited_by_in_set,
                share=gap.share,
                title=title,
                year=year,
            )
        )
    return tuple(out)


def facet_gaps(
    nodes: Sequence[Mapping[str, object]],
    *,
    thin_share: float = DEFAULT_THIN_SHARE,
) -> tuple[FacetGap, ...]:
    """Approach axes that are absent or thinly represented for this question.

    ``unclassified`` is never reported as a gap: it records that the classifier
    could not read the paper, which is a fact about the metadata rather than
    about the literature.
    """
    total = len(nodes)
    if not total:
        return ()
    held: Counter[str] = Counter()
    for node in nodes:
        for facet in node_facets(node):
            if facet != UNCLASSIFIED:
                held[facet] += 1

    known = [f for f in FACET_LABELS if f != UNCLASSIFIED]
    thin = [
        FacetGap(
            facet=facet,
            papers=held.get(facet, 0),
            share=held.get(facet, 0) / total,
            absent=held.get(facet, 0) == 0,
        )
        for facet in known
        if held.get(facet, 0) / total <= thin_share
    ]
    thin.sort(key=lambda gap: (gap.papers, gap.facet))
    return tuple(thin)


def gap_rows(gaps: Sequence[FoundationalGap]) -> list[list[object]]:
    """Serialisable rows for the graph document."""
    return [
        [gap.openalex_id, gap.cited_by_in_set, round(gap.share, 4), gap.title, gap.year]
        for gap in gaps
    ]


def read_gaps(document: Mapping[str, object]) -> tuple[FoundationalGap, ...]:
    """Parse persisted gaps, tolerating graphs harvested before they existed."""
    raw = document.get("foundational_gaps")
    if not isinstance(raw, list):
        return ()
    out = []
    for row in raw:
        if isinstance(row, list) and len(row) >= 3:
            title = row[3] if len(row) > 3 and isinstance(row[3], str) else None
            year = row[4] if len(row) > 4 and isinstance(row[4], int) else None
            out.append(
                FoundationalGap(
                    openalex_id=str(row[0]),
                    cited_by_in_set=int(row[1]) if isinstance(row[1], int) else 0,
                    share=float(row[2]) if isinstance(row[2], (int, float)) else 0.0,
                    title=title,
                    year=year,
                )
            )
    return tuple(out)
