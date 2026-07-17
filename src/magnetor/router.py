"""Cross-Domain Semantic Router (Spec Section 5) and retrieval orchestration.

A query is embedded once and scored against every domain by how well its most
relevant passages match — the mean cosine of the domain's top-K nearest
abstracts (Spec Section 5's "small representative sample"). It routes to the
single best domain, or the top two when their scores fall within a narrow margin
(the interdisciplinary signal). Scoring reuses each domain's own vector search,
so the router holds no document bodies and moves nothing across domains,
honouring the isolation invariant ("cross-domain awareness exists only at the
router layer").

A representative sample beats a single averaged centroid for multi-topic
domains: a centroid gets dragged toward a domain's bulk and under-scores a query
that matches a small but highly relevant cluster.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from magnetor.embeddings.base import Embedder
from magnetor.types import Domain
from magnetor.vectors import Vector, VectorIndex

#: Global (cross-domain) routing-decision log filename.
ROUTING_LOG_FILENAME = "routing_log.jsonl"
#: Default cosine margin within which a second domain is also selected.
DEFAULT_MARGIN = 0.05
#: How many top passages define a domain's representative match score.
DEFAULT_SAMPLE_SIZE = 3


@dataclass(frozen=True, slots=True)
class DomainScore:
    domain: Domain
    score: float


@dataclass(frozen=True, slots=True)
class Routing:
    """A routing decision: every domain's score and the selected top 1-2."""

    scored: tuple[DomainScore, ...]
    selected: tuple[Domain, ...]


@dataclass(frozen=True, slots=True)
class Hit:
    domain: Domain
    external_id: str
    score: float


class CrossDomainRouter:
    """Scores a query vector against per-domain centroids and selects domains."""

    def __init__(
        self,
        indices: dict[Domain, VectorIndex],
        *,
        margin: float = DEFAULT_MARGIN,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        log_path: Path | None = None,
    ) -> None:
        self._indices = indices
        self._margin = margin
        self._sample_size = max(1, sample_size)
        self._log_path = log_path

    @property
    def indices(self) -> dict[Domain, VectorIndex]:
        return self._indices

    def route(self, query_vector: Vector, *, query: str = "") -> Routing:
        """Score every non-empty domain by its top-K match mean and select 1-2."""
        scored: list[DomainScore] = []
        for domain, index in self._indices.items():
            hits = index.search(query_vector, self._sample_size)
            if not hits:  # domain has no embedded records yet
                continue
            score = sum(score for _, score in hits) / len(hits)
            scored.append(DomainScore(domain, score))
        scored.sort(key=lambda s: s.score, reverse=True)
        routing = Routing(tuple(scored), tuple(self._select(scored)))
        if self._log_path is not None:
            self._log(query, routing)
        return routing

    def _select(self, scored: list[DomainScore]) -> list[Domain]:
        if not scored:
            return []
        selected = [scored[0].domain]
        # Add the runner-up only if it is within the margin (top-2 max).
        if len(scored) > 1 and scored[0].score - scored[1].score <= self._margin:
            selected.append(scored[1].domain)
        return selected

    def _log(self, query: str, routing: Routing) -> None:
        assert self._log_path is not None
        entry = {
            "timestamp": dt.datetime.now(tz=dt.UTC).isoformat(),
            "query": query,
            "scored": [
                {"domain": s.domain.value, "score": round(s.score, 4)} for s in routing.scored
            ],
            "selected": [d.value for d in routing.selected],
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def retrieve(
    embedder: Embedder,
    router: CrossDomainRouter,
    query: str,
    *,
    k: int,
    domain: Domain | None = None,
) -> tuple[Routing | None, list[Hit]]:
    """Embed the query once, route it (unless ``domain`` is forced), and search.

    Returns the routing decision (``None`` when a domain was forced) and the
    hits across the selected domain(s), most similar first.
    """
    query_vector = embedder.embed_query(query)
    if domain is not None:
        routing: Routing | None = None
        selected: Sequence[Domain] = [domain]
    else:
        routing = router.route(query_vector, query=query)
        selected = routing.selected

    hits: list[Hit] = []
    for selected_domain in selected:
        index = router.indices.get(selected_domain)
        if index is None:
            continue
        for external_id, score in index.search(query_vector, k):
            hits.append(Hit(selected_domain, external_id, score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return routing, hits
