"""Branch C · L3.0 core — query-relative influence scoring (see ADR-0006).

Skeleton scope: the two uncontroversial core metrics only —

- **in-subgraph in-degree** — the "repetition" heuristic: how many papers *in
  the harvested set* cite this one. Global citation count is deliberately NOT
  used (ADR-0006 L3.0: influence is query-relative, not popularity).
- **PageRank** over the citation graph — structural importance within the
  lineage. Edges point citing -> cited, so rank accrues to cited work; no
  reversal needed.

Both are normalised to **percentile rank within the harvested set** so they are
distribution-free and comparable. The full multi-signal weighting profile
(ADR-0006 L3.1, versioned profile file) is deferred until this skeleton is
validated against real graphs — the composite here is a single principled metric,
not the final scheme.

PageRank is hand-rolled on NumPy rather than pulling in networkx for one
algorithm (escalation ladder, ADR-0001).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from magnetor.harvest import HarvestResult

_DAMPING = 0.85
_MAX_ITER = 100
_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class ScoredPaper:
    openalex_id: str
    in_degree: int
    in_degree_pct: float  # percentile rank within the set (0..1)
    pagerank: float
    pagerank_pct: float
    #: Skeleton influence = PageRank percentile (principled centrality). The full
    #: weighted composite (L3.1) arrives once the profile scheme is signed off.
    influence: float


@dataclass(frozen=True, slots=True)
class GraphScores:
    scored: tuple[ScoredPaper, ...]  # sorted by influence, descending


def score_graph(result: HarvestResult) -> GraphScores:
    """Score every harvested paper by in-degree and PageRank (percentile-normalised)."""
    ids = [p.openalex_id for p in result.papers]
    if not ids:
        return GraphScores(())

    in_degree = _in_degrees(ids, result.edges)
    pagerank = _pagerank(ids, result.edges)

    indeg_vec = np.array([in_degree[i] for i in ids], dtype=float)
    pr_vec = np.array([pagerank[i] for i in ids], dtype=float)
    indeg_pct = _percentile_ranks(indeg_vec)
    pr_pct = _percentile_ranks(pr_vec)

    scored = [
        ScoredPaper(
            openalex_id=nid,
            in_degree=in_degree[nid],
            in_degree_pct=float(indeg_pct[k]),
            pagerank=pagerank[nid],
            pagerank_pct=float(pr_pct[k]),
            influence=float(pr_pct[k]),
        )
        for k, nid in enumerate(ids)
    ]
    scored.sort(key=lambda s: s.influence, reverse=True)
    return GraphScores(tuple(scored))


def _in_degrees(ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, int]:
    """Count in-set citations received (edge is citing -> cited)."""
    degree = dict.fromkeys(ids, 0)
    for _citing, cited in edges:
        if cited in degree:
            degree[cited] += 1
    return degree


def _pagerank(ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Power-iteration PageRank; dangling nodes redistribute uniformly."""
    n = len(ids)
    index = {nid: i for i, nid in enumerate(ids)}
    out_targets: dict[int, list[int]] = defaultdict(list)
    out_degree = np.zeros(n)
    for citing, cited in edges:
        if citing in index and cited in index:
            out_targets[index[citing]].append(index[cited])
            out_degree[index[citing]] += 1

    rank = np.full(n, 1.0 / n)
    teleport = (1.0 - _DAMPING) / n
    for _ in range(_MAX_ITER):
        nxt = np.full(n, teleport)
        dangling = float(rank[out_degree == 0].sum())
        nxt += _DAMPING * dangling / n
        for i, targets in out_targets.items():
            share = _DAMPING * rank[i] / out_degree[i]
            for j in targets:
                nxt[j] += share
        if float(np.abs(nxt - rank).sum()) < _TOL:
            rank = nxt
            break
        rank = nxt
    return {nid: float(rank[index[nid]]) for nid in ids}


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Fraction of entries <= each entry (tie-aware, distribution-free)."""
    n = len(values)
    if n == 0:
        return values
    return np.array([float((values <= v).sum()) / n for v in values])
