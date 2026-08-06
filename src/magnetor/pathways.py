"""Branch C · L5.2 + L5.3 — pathway highlighting and traceable progression paths.

Two navigational features layered over an *already built* Evidence Graph
(ADR-0006). Both are computed server-side on submit and returned as plain node /
edge selections, so the existing static Graphviz renderer keeps working — no
interactive graph component is required (ADR-0006 L5.3).

- **L5.2 — highlighting.** A second, graph-scoped query seeds a Personalised
  PageRank (random-walk-with-restart): the L3.0 power iteration with a
  *non-uniform* teleport vector. High-scoring nodes plus the citation edges
  between them are the highlighted region. Structure alone cannot do this —
  in-degree/PageRank are query-blind and would return the same globally central
  papers for every sub-query — so the query enters through a content signal.

- **L5.3 — progression path.** One ordered chain with ranked alternatives
  ("Next"). Operator-confirmed **two-way**: ``roots ← anchor → development``, so
  a single path shows both what the anchor was built on and what grew out of it.
  Each step is tagged with its leg so the renderer can colour the two directions
  differently and the pathway can be traced by eye.

**Relevance is lexical, not embedded.** ADR-0006 L5.2 records that a term match
suffices for v1 ("free, no API, no key"); embeddings only upgrade the seed.
*Honest limitation:* a persisted node carries title and venue but deliberately
**not** the abstract (§3/I4 — graph artifacts hold identifiers, metrics and edges
only), so matching is title-scoped and therefore thinner than the "title+
abstract" the ADR assumed. Synonyms are missed; that is the documented cost of
keeping abstracts out of the artifact.

Nothing here is authoritative. Scores rank *likely* relevance over the harvested
slice, and the harvest is incomplete toward the roots (L8 boundary leakage), so
an "origin" is the earliest relevant paper *in available data* — never the
field's real origin. The L5.1(3) disclaimer applies to every output.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

# Mirrors graph_scoring: same damping and convergence discipline, since L5.2 is
# deliberately the same engine with a seeded teleport vector.
_DAMPING = 0.85
_MAX_ITER = 100
_TOL = 1e-9

#: Blend between keyword relevance and influence when weighting a step (L5.3.2).
DEFAULT_ALPHA = 0.5
#: Relevance a paper needs to be an *anchor* candidate (L5.3.1 — "the threshold
#: is a knob": too low and a marginally-relevant ancient paper wins the origin).
DEFAULT_ANCHOR_THRESHOLD = 0.34
#: Step weight a candidate must clear for the walk to continue (endpoint rule).
DEFAULT_FLOOR = 0.05
DEFAULT_MAX_DEPTH = 8
DEFAULT_K = 5

#: Leg tags. A step either runs back toward foundations or forward into later work.
ROOTS = "roots"
DEVELOPMENT = "development"

_WORD = re.compile(r"[a-z0-9]+")

# Function words carry no topical signal and would inflate every title's match
# fraction. Deliberately small: an aggressive list starts discarding real terms
# (e.g. "state", "field") that matter in physics titles.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "by",
        "from", "at", "as", "is", "are", "be", "was", "were", "its", "their", "this",
        "that", "these", "those", "we", "our", "using", "used", "use", "via", "into",
        "over", "under", "between", "about", "new", "novel", "toward", "towards",
    }
)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """The subset of a persisted node this layer needs (no abstract — §3/I4)."""

    id: str
    title: str
    year: int | None
    influence: float


@dataclass(frozen=True, slots=True)
class GraphView:
    """A typed read of a graph document. Edges point citing -> cited (new -> old)."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[str, str], ...]

    def ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    def by_id(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}


@dataclass(frozen=True, slots=True)
class Highlight:
    """L5.2 result: per-node relevance-seeded scores, plus the edges among winners."""

    scores: dict[str, float]  # Personalised PageRank, rescaled to 0..1
    relevance: dict[str, float]  # raw lexical seed, kept for audit (C1)
    selected: tuple[str, ...]  # nodes above the cut, score-descending
    edges: tuple[tuple[str, str], ...]  # citation edges with both ends selected


@dataclass(frozen=True, slots=True)
class PathStep:
    source: str
    target: str
    leg: str  # ROOTS or DEVELOPMENT


@dataclass(frozen=True, slots=True)
class ProgressionPath:
    """L5.3 result: one ordered two-way chain through the anchor."""

    anchor: str
    nodes: tuple[str, ...]  # oldest root … anchor … latest development
    steps: tuple[PathStep, ...]
    score: float  # sum of log step probabilities (higher = more probable)


def graph_view(document: Mapping[str, object]) -> GraphView:
    """Read a persisted graph document into a typed view, ignoring unknown fields."""
    raw_nodes = document.get("nodes")
    nodes: list[GraphNode] = []
    if isinstance(raw_nodes, list):
        for entry in raw_nodes:
            if not isinstance(entry, dict):
                continue
            nid = str(entry.get("id") or "")
            if not nid:
                continue
            nodes.append(
                GraphNode(
                    id=nid,
                    title=str(entry.get("title") or ""),
                    year=_as_year(entry.get("year")),
                    influence=_as_float(entry.get("influence")),
                )
            )
    known = {node.id for node in nodes}
    raw_edges = document.get("edges")
    edges: list[tuple[str, str]] = []
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if isinstance(edge, list) and len(edge) == 2:
                citing, cited = str(edge[0]), str(edge[1])
                # Drop dangling and self edges: a self-citation is an upstream data
                # error (already recorded by the harvest) and would trap a walk.
                if citing in known and cited in known and citing != cited:
                    edges.append((citing, cited))
    return GraphView(nodes=tuple(nodes), edges=tuple(edges))


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_year(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _terms(text: str) -> set[str]:
    """Content words of a string, lowercased; stopwords and 1-2 char tokens dropped."""
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def relevance_vector(view: GraphView, query: str) -> dict[str, float]:
    """Fraction of the sub-query's content words present in each node's title.

    Title-scoped by necessity (see module docstring). Returns 0.0 for every node
    when the query has no usable terms, which callers treat as "no seed".
    """
    wanted = _terms(query)
    if not wanted:
        return dict.fromkeys(view.ids(), 0.0)
    return {
        node.id: len(wanted & _terms(node.title)) / len(wanted) for node in view.nodes
    }


def personalised_pagerank(
    view: GraphView,
    seed: Mapping[str, float],
    *,
    damping: float = _DAMPING,
) -> dict[str, float]:
    """PageRank with teleport mass concentrated on the seeded (relevant) nodes.

    Identical iteration to ``graph_scoring._pagerank`` except the restart vector
    is the normalised seed instead of uniform — that single change is what makes
    the ranking query-aware (L5.2). With an all-zero seed it degenerates to the
    plain uniform PageRank, which is the honest fallback for an unmatched query.
    """
    ids = view.ids()
    n = len(ids)
    if n == 0:
        return {}
    index = {nid: i for i, nid in enumerate(ids)}

    teleport = np.array([max(0.0, float(seed.get(nid, 0.0))) for nid in ids])
    total = float(teleport.sum())
    teleport = np.full(n, 1.0 / n) if total <= 0.0 else teleport / total

    out_targets: dict[int, list[int]] = defaultdict(list)
    out_degree = np.zeros(n)
    for citing, cited in view.edges:
        out_targets[index[citing]].append(index[cited])
        out_degree[index[citing]] += 1

    rank: NDArray[np.float64] = teleport.copy()
    for _ in range(_MAX_ITER):
        nxt = (1.0 - damping) * teleport
        # Dangling mass returns to the seed, not uniformly: it must not leak
        # relevance away from the query.
        dangling = float(rank[out_degree == 0].sum())
        nxt = nxt + damping * dangling * teleport
        for i, targets in out_targets.items():
            share = damping * rank[i] / out_degree[i]
            for j in targets:
                nxt[j] += share
        if float(np.abs(nxt - rank).sum()) < _TOL:
            rank = nxt
            break
        rank = nxt
    return {nid: float(rank[index[nid]]) for nid in ids}


def highlight(
    view: GraphView,
    query: str,
    *,
    top_n: int = 15,
) -> Highlight:
    """L5.2: score every node against a graph-scoped sub-query and pick a region."""
    relevance = relevance_vector(view, query)
    raw = personalised_pagerank(view, relevance)
    peak = max(raw.values(), default=0.0)
    scores = {nid: (value / peak if peak > 0 else 0.0) for nid, value in raw.items()}

    ranked = sorted(scores, key=lambda nid: scores[nid], reverse=True)
    selected = tuple(ranked[:top_n])
    chosen = set(selected)
    edges = tuple((u, v) for u, v in view.edges if u in chosen and v in chosen)
    return Highlight(scores=scores, relevance=relevance, selected=selected, edges=edges)


def step_weights(
    view: GraphView, relevance: Mapping[str, float], *, alpha: float
) -> dict[str, float]:
    """``w(v) = a*keyword_relevance(v) + (1-a)*influence(v)`` (L5.3.2, ``a`` is a slider)."""
    alpha = min(1.0, max(0.0, alpha))
    return {
        node.id: alpha * float(relevance.get(node.id, 0.0)) + (1.0 - alpha) * node.influence
        for node in view.nodes
    }


def choose_anchor(
    view: GraphView,
    relevance: Mapping[str, float],
    *,
    threshold: float = DEFAULT_ANCHOR_THRESHOLD,
) -> str | None:
    """L5.3.1: the **earliest** paper among those relevant enough to the keyword.

    Two-factor, in this order: filter by relevance threshold, then take the
    oldest survivor. Not "the earliest paper in the graph" and not "the strongest
    match" — the operator's clarified rule. Undated papers cannot be ordered
    chronologically and so are only used when nothing dated qualifies.
    """
    candidates = [n for n in view.nodes if float(relevance.get(n.id, 0.0)) >= threshold]
    if not candidates:
        return None
    dated = [n for n in candidates if n.year is not None]
    pool = dated or candidates
    # Ties broken by relevance then influence, so the pick is deterministic.
    best = min(
        pool,
        key=lambda n: (
            n.year if n.year is not None else 1 << 30,
            -float(relevance.get(n.id, 0.0)),
            -n.influence,
        ),
    )
    return best.id


def _adjacency(edges: Sequence[tuple[str, str]]) -> tuple[
    dict[str, list[str]], dict[str, list[str]]
]:
    """Split citing->cited edges into the two traversal directions.

    ``roots[u]`` = what *u* cites (older, foundational). ``development[v]`` =
    what cites *v* (newer, built on it) — the reversed traversal L5.3 requires
    for a forward-in-time progression.
    """
    roots: dict[str, list[str]] = defaultdict(list)
    development: dict[str, list[str]] = defaultdict(list)
    for citing, cited in edges:
        roots[citing].append(cited)
        development[cited].append(citing)
    return roots, development


def _k_best(
    start: str,
    adjacency: Mapping[str, list[str]],
    weights: Mapping[str, float],
    *,
    k: int,
    floor: float,
    depth: int,
    memo: dict[tuple[str, int], list[tuple[float, tuple[str, ...]]]],
) -> list[tuple[float, tuple[str, ...]]]:
    """Up to ``k`` most-probable continuations from ``start``, as (logprob, nodes).

    Step probabilities are the candidate weights normalised over the *surviving*
    candidates, and a path score is their product — carried as a sum of logs to
    stay numerically sane. The walk is **open-ended** (the confirmed endpoint
    rule): it does not get to stop early, it stops only when no candidate clears
    the floor or the depth budget runs out. Were early stopping allowed, the
    empty path would always win, since every extra step multiplies by p < 1.

    Memoising on (node, depth) also serves as the cycle guard the ADR asks for:
    depth strictly decreases, so a genuine cycle cannot recurse forever.
    """
    key = (start, depth)
    cached = memo.get(key)
    if cached is not None:
        return cached

    stop: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    if depth <= 0:
        memo[key] = stop
        return stop

    candidates = [v for v in adjacency.get(start, ()) if weights.get(v, 0.0) >= floor]
    if not candidates:
        memo[key] = stop
        return stop

    total = sum(weights.get(v, 0.0) for v in candidates)
    if total <= 0.0:
        memo[key] = stop
        return stop

    # Guard against self-reference in the memo while this state is being solved:
    # a cycle would otherwise re-enter (start, depth) at the same depth.
    memo[key] = stop
    out: list[tuple[float, tuple[str, ...]]] = []
    for target in candidates:
        step = math.log(weights.get(target, 0.0) / total)
        for score, suffix in _k_best(
            target, adjacency, weights, k=k, floor=floor, depth=depth - 1, memo=memo
        ):
            if target in suffix:  # rare genuine cycle — drop rather than loop
                continue
            out.append((step + score, (target, *suffix)))
    if not out:
        memo[key] = stop
        return stop
    out.sort(key=lambda item: item[0], reverse=True)
    trimmed = out[:k]
    memo[key] = trimmed
    return trimmed


def progression_paths(
    view: GraphView,
    query: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_ANCHOR_THRESHOLD,
    floor: float = DEFAULT_FLOOR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    k: int = DEFAULT_K,
) -> tuple[ProgressionPath, ...]:
    """L5.3: ranked two-way progression paths ``roots ← anchor → development``.

    The anchor is the earliest sufficiently-relevant paper (L5.3.1). From it the
    walk runs in *both* directions, as confirmed by the operator: back through
    what it was built on and forward through what built on it. Alternatives for
    the "Next" control are the best combinations of the two legs, ranked by joint
    log-probability.

    The floor is applied to the **blended** step weight rather than raw keyword
    relevance, because foundational papers are rarely keyword-matches — a raw
    relevance floor would sever the roots leg, which is the half the operator
    specifically asked for. Slide ``alpha`` toward influence to lengthen that leg.
    """
    relevance = relevance_vector(view, query)
    anchor = choose_anchor(view, relevance, threshold=threshold)
    if anchor is None:
        return ()

    weights = step_weights(view, relevance, alpha=alpha)
    roots_adj, dev_adj = _adjacency(view.edges)

    back = _k_best(
        anchor, roots_adj, weights, k=k, floor=floor, depth=max_depth, memo={}
    )
    forward = _k_best(
        anchor, dev_adj, weights, k=k, floor=floor, depth=max_depth, memo={}
    )

    combined: list[ProgressionPath] = []
    for back_score, back_nodes in back:
        for fwd_score, fwd_nodes in forward:
            if set(back_nodes) & set(fwd_nodes):  # never reuse a node on both legs
                continue
            steps = [PathStep(source=anchor, target=t, leg=ROOTS) for t in back_nodes[:1]]
            steps += [
                PathStep(source=back_nodes[i], target=back_nodes[i + 1], leg=ROOTS)
                for i in range(len(back_nodes) - 1)
            ]
            steps += [PathStep(source=anchor, target=t, leg=DEVELOPMENT) for t in fwd_nodes[:1]]
            steps += [
                PathStep(source=fwd_nodes[i], target=fwd_nodes[i + 1], leg=DEVELOPMENT)
                for i in range(len(fwd_nodes) - 1)
            ]
            combined.append(
                ProgressionPath(
                    anchor=anchor,
                    # Oldest first: the roots leg walks backwards in time, so it
                    # reads correctly only when reversed.
                    nodes=(*reversed(back_nodes), anchor, *fwd_nodes),
                    steps=tuple(steps),
                    score=back_score + fwd_score,
                )
            )
    combined.sort(key=lambda p: p.score, reverse=True)
    return tuple(combined[:k])


#: How a selected pair turns out to be related.
LINEAGE = "lineage"  # a directed citation chain exists: one descends from the other
INDIRECT = "indirect"  # joined only by ignoring citation direction
UNCONNECTED = "none"

#: Leg tag for edges on a checked connection, so the renderer can colour them
#: apart from a traced progression.
LINK = "link"


@dataclass(frozen=True, slots=True)
class PairLink:
    """How two selected papers relate, with the chain that shows it."""

    a: str
    b: str
    kind: str  # LINEAGE | INDIRECT | UNCONNECTED
    path: tuple[str, ...]  # ordered, from a to b; empty when unconnected
    edges: tuple[tuple[str, str], ...]  # the chain in stored citing->cited form


@dataclass(frozen=True, slots=True)
class ConnectionReport:
    pairs: tuple[PairLink, ...]
    nodes: tuple[str, ...]  # every node on any chain found
    edges: dict[tuple[str, str], str]  # renderer overlay: edge -> LINK
    all_connected: bool


def _bfs(adjacency: Mapping[str, set[str]], start: str, goal: str) -> tuple[str, ...] | None:
    """Shortest chain from ``start`` to ``goal``, or ``None``. Breadth-first, so
    the first chain reached is a shortest one."""
    if start == goal:
        return (start,)
    previous: dict[str, str] = {start: start}
    queue = [start]
    while queue:
        nxt: list[str] = []
        for node in queue:
            for neighbour in adjacency.get(node, ()):
                if neighbour in previous:
                    continue
                previous[neighbour] = node
                if neighbour == goal:
                    chain = [goal]
                    while chain[-1] != start:
                        chain.append(previous[chain[-1]])
                    return tuple(reversed(chain))
                nxt.append(neighbour)
        queue = nxt
    return None


def connect(view: GraphView, selected: Sequence[str]) -> ConnectionReport:
    """Check whether selected papers are connected, and by what chain.

    Three answers, because in a citation DAG "connected" is genuinely ambiguous
    and the distinction is the useful part. Measured on this project's corpus,
    only 7.5-34% of ordered pairs have a directed path and some graphs split into
    four components, so both weaker answers occur often enough to matter:

    - **lineage** — a directed citation chain runs one way, so one paper descends
      from the other. The strong answer.
    - **indirect** — joined only once direction is ignored: they share a relative
      but neither descends from the other.
    - **none** — different components of this graph. Note the honest reading:
      unconnected *in the harvested slice*, not unrelated in the literature.
    """
    directed: dict[str, set[str]] = defaultdict(set)
    undirected: dict[str, set[str]] = defaultdict(set)
    stored = set(view.edges)
    for citing, cited in view.edges:
        directed[citing].add(cited)
        undirected[citing].add(cited)
        undirected[cited].add(citing)

    def orient(x: str, y: str) -> tuple[str, str]:
        return (x, y) if (x, y) in stored else (y, x)

    pairs: list[PairLink] = []
    touched: set[str] = set()
    overlay: dict[tuple[str, str], str] = {}
    for a, b in combinations(list(dict.fromkeys(selected)), 2):
        chain = _bfs(directed, a, b)
        kind = LINEAGE
        if chain is None:
            reverse = _bfs(directed, b, a)
            chain = tuple(reversed(reverse)) if reverse else None
        if chain is None:
            chain = _bfs(undirected, a, b)
            kind = INDIRECT if chain else UNCONNECTED
        if not chain:
            pairs.append(PairLink(a=a, b=b, kind=UNCONNECTED, path=(), edges=()))
            continue
        steps = tuple(orient(chain[i], chain[i + 1]) for i in range(len(chain) - 1))
        pairs.append(PairLink(a=a, b=b, kind=kind, path=chain, edges=steps))
        touched.update(chain)
        for step in steps:
            overlay[step] = LINK
    return ConnectionReport(
        pairs=tuple(pairs),
        nodes=tuple(touched),
        edges=overlay,
        all_connected=bool(pairs) and all(p.kind != UNCONNECTED for p in pairs),
    )


def path_edges(path: ProgressionPath) -> dict[tuple[str, str], str]:
    """Map each traced step onto its stored citing->cited edge, keeping the leg tag.

    A development step runs cited->citing (the reversed traversal), so it is
    flipped back to the stored orientation for the renderer to find it.
    """
    mapping: dict[tuple[str, str], str] = {}
    for step in path.steps:
        if step.leg == ROOTS:
            mapping[(step.source, step.target)] = ROOTS
        else:
            mapping[(step.target, step.source)] = DEVELOPMENT
    return mapping
