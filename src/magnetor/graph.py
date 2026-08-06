"""Branch C · L4 — evidence-graph assembly and persistence (see ADR-0006).

Joins the three layers — harvested papers (L1), influence scores (L3.0), and rank
robustness (L8) — into one serialisable graph document, persisted under
``<data-root>/graphs/<query_hash>.json``. The dashboard only ever *reads* these
documents; building them is the offline ``magnetor harvest`` batch job (D1).

A node carries what a researcher needs to trace the pathway: identity + link,
the query-relative influence, the rank confidence interval, and the integrity
flag (retraction). Contrarian typing (L3.0a) and the full weighted composite
(L3.1) are deferred with the rest of the metric catalogue — the skeleton node is
deliberately honest about carrying only the two core metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from magnetor.config import global_store_path
from magnetor.graph_scoring import GraphScores
from magnetor.harvest import HarvestedPaper, HarvestResult
from magnetor.relations import DerivedRelations, RelationEdge
from magnetor.robustness import Robustness

GRAPHS_DIR = "graphs"


def _node_url(paper: HarvestedPaper) -> str:
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    return f"https://openalex.org/{paper.openalex_id}"


def build_graph_document(
    result: HarvestResult,
    scores: GraphScores,
    robustness: Robustness,
    *,
    top_n: int | None = None,
    relations: DerivedRelations | None = None,
) -> dict[str, Any]:
    """Merge papers + scores + rank intervals into one graph document.

    ``relations`` adds the derived co-citation / bibliographic-coupling layers
    (ADR-0006 L4). They are stored under their own keys and never merged into
    ``edges``, which stays the real citation backbone (I2). Omitting them writes a
    document identical to before, so graphs harvested earlier still load.
    """
    paper_by_id = {p.openalex_id: p for p in result.papers}
    score_by_id = {s.openalex_id: s for s in scores.scored}
    interval_by_id = {r.openalex_id: r for r in robustness.intervals}

    ordered = [s.openalex_id for s in scores.scored]  # already influence-desc
    if top_n is not None:
        ordered = ordered[:top_n]
    keep = set(ordered)

    nodes: list[dict[str, Any]] = []
    for nid in ordered:
        paper = paper_by_id[nid]
        score = score_by_id[nid]
        interval = interval_by_id.get(nid)
        nodes.append(
            {
                "id": nid,
                "title": paper.title,
                "year": paper.year,
                "doi": paper.doi,
                "url": _node_url(paper),
                "venue": paper.venue,
                "in_degree": score.in_degree,
                "pagerank": round(score.pagerank, 6),
                "influence": round(score.influence, 4),
                "median_rank": interval.median_rank if interval else None,
                "lo_rank": interval.lo_rank if interval else None,
                "hi_rank": interval.hi_rank if interval else None,
                "stable": interval.stable if interval else None,
                "is_retracted": paper.is_retracted,
                "is_review": paper.is_review,
            }
        )
    edges = [[u, v] for u, v in result.edges if u in keep and v in keep]
    document: dict[str, Any] = {
        "query": result.query,
        "generated_at": result.generated_at,
        "n_fetched": result.n_fetched,
        "boundary_leakage": round(robustness.boundary_leakage, 4),
        "resamples": robustness.resamples,
        # Durable record of self-citation edges dropped (upstream data errors).
        "self_citations_removed": list(result.self_referencing_ids),
        "nodes": nodes,
        "edges": edges,
    }
    if relations is not None:
        document["biblio_coupled"] = _relation_rows(relations.biblio_coupled, keep)
        document["co_cited"] = _relation_rows(relations.co_cited, keep)
    return document


def _relation_rows(
    edges: Sequence[RelationEdge], keep: set[str]
) -> list[list[str | int]]:
    """Derived edges as ``[source, target, weight]``, pruned to the kept nodes."""
    return [
        [e.source, e.target, e.weight]
        for e in edges
        if e.source in keep and e.target in keep
    ]


def query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _graphs_root(graphs_dir: Path | None) -> Path:
    return graphs_dir or global_store_path(GRAPHS_DIR)


def save_graph(document: dict[str, Any], *, graphs_dir: Path | None = None) -> Path:
    root = _graphs_root(graphs_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{query_hash(str(document.get('query', '')))}.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def load_graph(query: str, *, graphs_dir: Path | None = None) -> dict[str, Any] | None:
    path = _graphs_root(graphs_dir) / f"{query_hash(query)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def list_graphs(*, graphs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Summaries of persisted graphs (query, when, size) for a picker, newest first."""
    root = _graphs_root(graphs_dir)
    if not root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict):
            summaries.append(
                {
                    "query": doc.get("query", ""),
                    "generated_at": doc.get("generated_at", ""),
                    "n_nodes": len(doc.get("nodes", [])),
                }
            )
    summaries.sort(key=lambda s: str(s.get("generated_at", "")), reverse=True)
    return summaries
