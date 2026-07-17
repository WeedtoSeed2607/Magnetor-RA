"""Semantic Scholar citation expansion (Spec Section 7.2, Anchor-Lock).

When Branch B locks a single anchor paper, the spec expands its
forward/backward citations "via the Semantic Scholar API". This module is that
boundary: given a stored :class:`Paper`, it fetches the papers that cite it
(forward) and the papers it cites (backward).

Citations are enrichment, not core: if Semantic Scholar rate-limits, 404s, or is
unreachable, expansion degrades to empty lists rather than failing the deep-dive.
Unauthenticated access is aggressively rate-limited, so requests are throttled
and an optional ``MAGNETOR_S2_API_KEY`` lifts the limit.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

from magnetor import _http
from magnetor.errors import SourceUnavailableError
from magnetor.types import Paper

_API_BASE = "https://api.semanticscholar.org/graph/v1"
_KEY_ENV = "MAGNETOR_S2_API_KEY"
#: Semantic Scholar allows ~100 requests / 5 min unauthenticated; be polite.
_REQUEST_INTERVAL = 1.0
#: Cap on how many citations/references to pull per edge.
_DEFAULT_LIMIT = 25
_FIELDS = "title,year,externalIds"
_ARXIV_VERSION = re.compile(r"v\d+$")


@dataclass(frozen=True, slots=True)
class Citation:
    """A single citing or cited paper. Grounded — sourced verbatim from S2."""

    title: str
    year: int | None
    doi: str | None = None
    arxiv_id: str | None = None


class SemanticScholarClient:
    """Fetch forward/backward citations for a paper from Semantic Scholar."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        throttle: _http.Throttle | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> None:
        self._api_key = api_key or os.environ.get(_KEY_ENV)
        self._client = client
        self._throttle = throttle or _http.Throttle(_REQUEST_INTERVAL)
        self._limit = limit

    def expand(self, paper: Paper) -> tuple[list[Citation], list[Citation]]:
        """Return ``(forward_citations, backward_references)`` for ``paper``.

        ``forward`` = papers that cite the anchor; ``backward`` = papers it cites.
        Empty lists if the paper has no resolvable id or S2 is unavailable.
        """
        paper_id = _paper_id(paper)
        if paper_id is None:
            return [], []
        forward = self._fetch(paper_id, "citations", "citingPaper")
        backward = self._fetch(paper_id, "references", "citedPaper")
        return forward, backward

    def _headers(self) -> dict[str, str] | None:
        return {"x-api-key": self._api_key} if self._api_key else None

    def _fetch(self, paper_id: str, edge: str, wrapper: str) -> list[Citation]:
        try:
            text = _http.get(
                f"{_API_BASE}/paper/{paper_id}/{edge}",
                params={"fields": _FIELDS, "limit": str(self._limit)},
                headers=self._headers(),
                client=self._client,
                throttle=self._throttle,
            )
        except SourceUnavailableError:
            # Not found / rate-limited / unreachable: citations are optional.
            return []
        return _parse_edge(text, wrapper)


def _paper_id(paper: Paper) -> str | None:
    """Map a stored paper to a Semantic Scholar id, preferring a DOI."""
    if paper.doi:
        return f"DOI:{paper.doi}"
    if paper.source == "arXiv" and paper.external_id:
        return f"ARXIV:{_ARXIV_VERSION.sub('', paper.external_id)}"
    if paper.external_id.startswith("PMC"):
        return f"PMCID:{paper.external_id}"
    return None


def _parse_edge(text: str, wrapper: str) -> list[Citation]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    citations: list[Citation] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        inner = item.get(wrapper)
        if isinstance(inner, dict):
            citations.append(_to_citation(inner))
    return citations


def _to_citation(paper: dict[str, object]) -> Citation:
    external = paper.get("externalIds")
    ids = external if isinstance(external, dict) else {}
    year = paper.get("year")
    return Citation(
        title=str(paper.get("title") or ""),
        year=year if isinstance(year, int) else None,
        doi=_opt_str(ids.get("DOI")),
        arxiv_id=_opt_str(ids.get("ArXiv")),
    )


def _opt_str(value: object) -> str | None:
    return str(value) if value else None
