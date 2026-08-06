"""Branch C · L4 — derived citation relations: co-citation and bibliographic coupling.

ADR-0006 L4 lists ``co_cited`` and ``biblio_coupled`` beside ``cites`` and notes
both are "derivable from data already fetched — no additional API calls". This
module computes them from the reference lists the harvest already holds in
memory, so nothing here touches the network.

**Why they earn their place.** Measured on this project's own corpus: direct
citation links **2.2%** of paper pairs, while sharing **>=3 references from
outside the harvested set** links **19.8%** — nearly nine times the structure,
recovered from exactly the references the boundary-leakage figure counts as lost.
Two papers that never cite one another but lean on the same earlier work are
related, and a citation backbone cannot say so. That is the computable form of
"which earlier work do these studies all lean on".

**Two constraints this module deliberately respects.**

- **D4 — the metric catalogue is gated.** These relations are *navigational*:
  they change what can be seen, never what a node scores. Influence remains
  in-subgraph PageRank over citations alone.
- **I2 — no synthetic edges in the backbone.** These are real bibliometric
  relations, not similarity guesses, but they are still *derived*, so they are
  persisted and drawn as their own layer and never merged with ``cites``.

**Honest limitation on co-citation.** Standard co-citation counts how often two
works are cited together by *any* later paper. Here the citing population is only
the harvested set, so the count is "cited together by papers in this harvest".
That understates co-citation for works whose citers were not harvested, and the
bias grows with boundary leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from magnetor.harvest import HarvestResult

#: Shared references two papers need before a coupling edge is drawn. Chosen from
#: the corpus measurement above: >=1 shared reference links 62.6% of pairs, which
#: is a hairball (R4); >=3 keeps ~20% and still swamps direct citation.
DEFAULT_MIN_SHARED = 3

#: Times two papers must be cited together before a co-citation edge is drawn.
DEFAULT_MIN_CO_CITATIONS = 2

#: Strongest edges kept per node, so one heavily-shared foundation cannot bury the
#: graph in a clique (ADR-0006 R4 — the stored graph stays complete, the drawn one
#: does not have to be).
DEFAULT_TOP_K = 6

#: References cited by more than this many papers are skipped when pairing. Such a
#: reference is a field-wide staple: it pairs everything with everything, costs
#: O(n^2) to expand, and carries almost no discriminating signal.
DEFAULT_MAX_FANOUT = 200


@dataclass(frozen=True, slots=True)
class RelationEdge:
    """An undirected derived relation. ``source`` < ``target`` for a stable key."""

    source: str
    target: str
    weight: int


@dataclass(frozen=True, slots=True)
class DerivedRelations:
    biblio_coupled: tuple[RelationEdge, ...]
    co_cited: tuple[RelationEdge, ...]


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _co_occurrences(
    groups: Mapping[str, frozenset[str]] | Mapping[str, set[str]],
    *,
    max_fanout: int,
) -> Counter[tuple[str, str]]:
    """Count how often two members co-occur across the given groups.

    Inverted index rather than all-pairs: for each shared item, pair up only the
    members that actually carry it. Cost is bounded by the fanout cap.
    """
    holders: dict[str, list[str]] = defaultdict(list)
    for member, items in groups.items():
        for item in items:
            holders[item].append(member)
    counts: Counter[tuple[str, str]] = Counter()
    for members in holders.values():
        if len(members) < 2 or len(members) > max_fanout:
            continue
        for a, b in combinations(sorted(members), 2):
            counts[(a, b)] += 1
    return counts


def _prune(
    counts: Counter[tuple[str, str]], *, minimum: int, top_k: int
) -> tuple[RelationEdge, ...]:
    """Drop weak pairs, then keep each node's strongest ``top_k``.

    An edge survives if *either* endpoint ranks it in its own top-k, so a paper
    with few relations keeps its best links even when its partner is popular.
    """
    kept = {pair: w for pair, w in counts.items() if w >= minimum}
    if not kept:
        return ()
    per_node: dict[str, list[tuple[int, tuple[str, str]]]] = defaultdict(list)
    for pair, weight in kept.items():
        per_node[pair[0]].append((weight, pair))
        per_node[pair[1]].append((weight, pair))
    survivors: set[tuple[str, str]] = set()
    for ranked in per_node.values():
        ranked.sort(key=lambda item: (-item[0], item[1]))
        survivors.update(pair for _weight, pair in ranked[:top_k])
    edges = [RelationEdge(source=a, target=b, weight=kept[(a, b)]) for a, b in survivors]
    edges.sort(key=lambda e: (-e.weight, e.source, e.target))
    return tuple(edges)


def bibliographic_coupling(
    references: Mapping[str, frozenset[str]],
    *,
    min_shared: int = DEFAULT_MIN_SHARED,
    top_k: int = DEFAULT_TOP_K,
    max_fanout: int = DEFAULT_MAX_FANOUT,
) -> tuple[RelationEdge, ...]:
    """Papers sharing references — relatedness by common ancestry (Kessler, 1963).

    Shared references are counted whether or not they were harvested: an ancestor
    outside the set is the *most* informative kind, since neither paper's link to
    it is visible in the citation backbone.
    """
    return _prune(
        _co_occurrences(references, max_fanout=max_fanout),
        minimum=min_shared,
        top_k=top_k,
    )


def co_citation(
    references: Mapping[str, frozenset[str]],
    *,
    min_co_citations: int = DEFAULT_MIN_CO_CITATIONS,
    top_k: int = DEFAULT_TOP_K,
    max_fanout: int = DEFAULT_MAX_FANOUT,
) -> tuple[RelationEdge, ...]:
    """Papers cited together — relatedness conferred by later authors (Small, 1973).

    Only pairs among *harvested* papers are counted, since a pair is only useful
    here if both ends can be drawn. See the module docstring for the resulting
    undercount.
    """
    known = set(references)
    # Invert the reference lists: co-citation pairs the works being *cited*, so
    # the members must be cited works and the shared items their citers. Pairing
    # the citing papers instead would silently recompute coupling.
    cited_by: dict[str, set[str]] = defaultdict(set)
    for citing, refs in references.items():
        for ref in refs & known:
            cited_by[ref].add(citing)
    return _prune(
        _co_occurrences(cited_by, max_fanout=max_fanout),
        minimum=min_co_citations,
        top_k=top_k,
    )


def reference_map(result: HarvestResult) -> dict[str, frozenset[str]]:
    """Reference lists per harvested paper, self-citations removed."""
    return {
        paper.openalex_id: frozenset(
            ref for ref in paper.referenced_works if ref != paper.openalex_id
        )
        for paper in result.papers
    }


def derive_relations(
    result: HarvestResult,
    *,
    min_shared: int = DEFAULT_MIN_SHARED,
    min_co_citations: int = DEFAULT_MIN_CO_CITATIONS,
    top_k: int = DEFAULT_TOP_K,
) -> DerivedRelations:
    """Both derived layers for a harvest, computed without any further API calls."""
    references = reference_map(result)
    return DerivedRelations(
        biblio_coupled=bibliographic_coupling(
            references, min_shared=min_shared, top_k=top_k
        ),
        co_cited=co_citation(references, min_co_citations=min_co_citations, top_k=top_k),
    )
