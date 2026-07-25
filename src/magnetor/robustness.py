"""Branch C · L8 — robustness layer (see ADR-0006 §L8).

The output is a *ranked* graph, so what must be demonstrated is not that a score
is "correct" but that the **ranking is stable under perturbation**. This module
carries the two cheapest, highest-value checks (Tier 3, pure computation, always
full strength):

- **Bootstrap rank CIs** — resample the graph many times and report each paper's
  rank as a distribution: "rank 3 (95% CI 2-7)" instead of a bare "3". Ranks
  whose interval spans a wide band are flagged unstable, reusing Branch A's
  LOW SUPPORT convention.
- **Boundary leakage** — the fraction of citations pointing *outside* the
  harvested set: a cheap completeness signal (high leakage ⇒ the boundary is
  cutting through the lineage; harvest wider).

:func:`kendall_tau` is the rank-correlation primitive the depth-convergence check
(PageRank at snowball depth k vs k+1) will use once the snowball harvest exists;
it is provided now so that check is a thin wrapper later.

Thresholds are deliberately provisional (ADR-0006 §L8): the "stable" band is a
labelled convention to calibrate during researcher testing, not a law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnetor.graph_scoring import score_graph
from magnetor.harvest import HarvestResult

DEFAULT_RESAMPLES = 1000
_DEFAULT_DROP_FRAC = 0.1
_CI = 0.95
#: Provisional: a paper is "stable" if its rank CI spans <= this fraction of n.
_STABLE_BAND_FRAC = 0.1


@dataclass(frozen=True, slots=True)
class RankInterval:
    openalex_id: str
    median_rank: float
    lo_rank: int  # best (smallest) rank at the CI's lower bound
    hi_rank: int  # worst (largest) rank at the CI's upper bound
    stable: bool  # CI band within the provisional _STABLE_BAND_FRAC of n


@dataclass(frozen=True, slots=True)
class Robustness:
    intervals: tuple[RankInterval, ...]  # ordered by median rank (best first)
    boundary_leakage: float  # fraction of citations leaving the harvested set
    resamples: int


def boundary_leakage(result: HarvestResult) -> float:
    """Fraction of all references that point *outside* the harvested set."""
    ids = {p.openalex_id for p in result.papers}
    total = 0
    external = 0
    for paper in result.papers:
        for ref in paper.referenced_works:
            total += 1
            if ref not in ids:
                external += 1
    return external / total if total else 0.0


def bootstrap_rank_cis(
    result: HarvestResult,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    drop_frac: float = _DEFAULT_DROP_FRAC,
    seed: int = 0,
) -> Robustness:
    """Resample the graph, re-score, and report each paper's rank distribution."""
    ids = [p.openalex_id for p in result.papers]
    n = len(ids)
    if n == 0:
        return Robustness((), 0.0, resamples)

    paper_by_id = {p.openalex_id: p for p in result.papers}
    rng = np.random.default_rng(seed)
    keep = max(1, round(n * (1.0 - drop_frac)))
    ranks: dict[str, list[int]] = {i: [] for i in ids}

    for _ in range(resamples):
        chosen = {ids[k] for k in rng.choice(n, size=keep, replace=False)}
        sub_edges = tuple((u, v) for u, v in result.edges if u in chosen and v in chosen)
        sub = HarvestResult(
            query=result.query,
            generated_at=result.generated_at,
            papers=tuple(paper_by_id[i] for i in chosen),
            edges=sub_edges,
            n_fetched=len(chosen),
        )
        for rank, scored in enumerate(score_graph(sub).scored, start=1):
            ranks[scored.openalex_id].append(rank)

    lo_q, hi_q = (1.0 - _CI) / 2.0, 1.0 - (1.0 - _CI) / 2.0
    band = max(1, round(_STABLE_BAND_FRAC * n))
    intervals: list[RankInterval] = []
    for nid in ids:
        samples = ranks[nid]
        if not samples:  # never survived a resample (vanishingly unlikely)
            continue
        arr = np.array(samples)
        lo = int(np.quantile(arr, lo_q))
        hi = int(np.quantile(arr, hi_q))
        intervals.append(
            RankInterval(
                openalex_id=nid,
                median_rank=float(np.median(arr)),
                lo_rank=lo,
                hi_rank=hi,
                stable=(hi - lo) <= band,
            )
        )
    intervals.sort(key=lambda r: r.median_rank)
    return Robustness(tuple(intervals), boundary_leakage(result), resamples)


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall rank correlation over the ids common to both orderings (-1..1).

    The primitive for depth-convergence (ADR-0006 §L8): when tau(depth_k,
    depth_k+1) exceeds the operator's threshold, the boundary no longer changes
    conclusions and expansion stops.
    """
    common = [i for i in order_a if i in set(order_b)]
    rank_b = {nid: k for k, nid in enumerate(order_b)}
    m = len(common)
    if m < 2:
        return 1.0
    concordant = discordant = 0
    for x in range(m):
        for y in range(x + 1, m):
            # order_a is already sorted best->worst by construction (x < y).
            b = rank_b[common[x]] - rank_b[common[y]]
            if b < 0:
                concordant += 1
            elif b > 0:
                discordant += 1
    pairs = m * (m - 1) // 2
    return (concordant - discordant) / pairs if pairs else 1.0
