"""Branch C — anchored mode: build a graph outward from **one paper**.

The existing harvest answers "what is the lineage of this *question*". This
answers the operator's second mode: "take a paper I already have and work out its
relationship to everything around it, to understand its evolution."

**Implemented as a ``WorksSource``, not as a change to the harvest.** ``run_harvest``
already accepts any source that can ``search`` and ``fetch_by_ids``; anchored mode
is therefore a source whose "search" returns a citation neighbourhood instead of
keyword hits. Everything downstream — the works cache, snowball expansion, edge
building, scoring, robustness, the derived relation layers, the dashboard and the
export — is inherited without modification.

**The two directions cost very different amounts, and the defaults reflect that.**
Backward is free: ``referenced_works`` arrives inline on every work already
fetched, so walking toward antecedents costs no extra request. Forward is not:
finding what cites a paper needs one query per paper, so forward expansion is
applied to the seed and then only to a bounded, most-cited slice of the first
hop. Treating the two symmetrically would multiply requests for a graph that is
mostly noise at the edges.

**Scope, stated rather than implied.** v1 is OpenAlex only. The operator asked for
"all of the APIs and archives"; Semantic Scholar, Crossref and PubPeer each have
their own identifier and rate-limit story, and folding them in here would make
one change that fails in four ways.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from magnetor import _http
from magnetor.errors import MagnetorError
from magnetor.harvest import (
    OPENALEX_BASE,
    HarvestResult,
    WorksSource,
    run_harvest,
)

#: Fields requested per work. Deliberately a local copy rather than an import of
#: the harvest's private constant: this module must keep working if that one is
#: retuned, and a missing field here degrades a node, it does not break a run.
_SELECT = (
    "id,display_name,title,publication_year,doi,cited_by_count,"
    "referenced_works,type,is_retracted,authorships,primary_location,"
    "abstract_inverted_index"
)

_MAX_PER_PAGE = 200
_ID_BATCH = 50

#: Hops walked toward antecedents. Cheap - references come inline.
DEFAULT_BACKWARD_HOPS = 2
#: First-hop papers that also get forward-expanded, most-cited first. 0 means the
#: seed alone is expanded forward, which is the honest default: one query.
DEFAULT_FORWARD_FANOUT = 0
#: Citing papers pulled per forward query.
DEFAULT_FORWARD_LIMIT = 100

_OPENALEX_ID = re.compile(r"\bW\d{5,}\b")
_DOI = re.compile(r"10\.\d{4,9}/\S+")


class AnchorError(MagnetorError):
    """The seed paper could not be resolved to a work."""


class NeighbourSource(Protocol):
    """Boundary for citation-neighbourhood lookups — OpenAlex live, fake in tests."""

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        ...

    def cited_by(self, work_id: str, *, limit: int) -> list[dict[str, Any]]:
        """Works that cite ``work_id`` — the forward direction."""
        ...

    def resolve(self, reference: str) -> dict[str, Any] | None:
        """Find one work from an OpenAlex id, a DOI, or a URL containing either."""
        ...


class OpenAlexNeighbours:
    """OpenAlex implementation of :class:`NeighbourSource` (free, no key)."""

    def __init__(
        self, *, mailto: str | None = None, throttle: _http.Throttle | None = None
    ) -> None:
        self._mailto = mailto or os.environ.get(
            "MAGNETOR_OPENALEX_MAILTO", "magnetor@example.com"
        )
        self._throttle = throttle if throttle is not None else _http.Throttle(0.1)

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        params = {**params, "mailto": self._mailto, "select": _SELECT}
        raw = _http.get(OPENALEX_BASE, params=params, throttle=self._throttle)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        unique = [i for i in dict.fromkeys(ids) if i]
        out: list[dict[str, Any]] = []
        for start in range(0, len(unique), _ID_BATCH):
            batch = unique[start : start + _ID_BATCH]
            data = self._get(
                {
                    "filter": "ids.openalex:" + "|".join(batch),
                    "per-page": str(_ID_BATCH),
                }
            )
            out.extend(data.get("results") or [])
        return out

    def cited_by(self, work_id: str, *, limit: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor: str | None = "*"
        while len(out) < limit and cursor:
            data = self._get(
                {
                    "filter": f"cites:{work_id}",
                    "per-page": str(min(_MAX_PER_PAGE, limit - len(out))),
                    "cursor": cursor,
                }
            )
            batch = data.get("results") or []
            out.extend(batch)
            if not batch:
                break
            nxt = (data.get("meta") or {}).get("next_cursor")
            cursor = nxt if isinstance(nxt, str) else None
        return out[:limit]

    def resolve(self, reference: str) -> dict[str, Any] | None:
        work_id = extract_openalex_id(reference)
        if work_id:
            found = self.fetch_by_ids([work_id])
            return found[0] if found else None
        doi = extract_doi(reference)
        if doi:
            data = self._get({"filter": f"doi:{doi}", "per-page": "1"})
            results = data.get("results") or []
            return results[0] if results else None
        return None


def extract_openalex_id(reference: str) -> str | None:
    """Pull a ``W…`` id out of a bare id or an openalex.org URL."""
    match = _OPENALEX_ID.search(reference.strip())
    return match.group(0) if match else None


def extract_doi(reference: str) -> str | None:
    """Pull a DOI out of a bare DOI or a doi.org URL, trailing punctuation removed."""
    match = _DOI.search(reference.strip())
    return match.group(0).rstrip(").,;") if match else None


def _short_id(url: object) -> str:
    text = str(url or "")
    return text.rsplit("/", 1)[-1] if text else ""


class AnchoredSource:
    """A :class:`~magnetor.harvest.WorksSource` returning a paper's neighbourhood.

    ``search`` ignores the notion of a keyword: the "query" it receives is the
    seed reference, and what comes back is the ego network around it. That is what
    lets the whole existing harvest pipeline run unchanged.
    """

    def __init__(
        self,
        neighbours: NeighbourSource,
        *,
        backward_hops: int = DEFAULT_BACKWARD_HOPS,
        forward_fanout: int = DEFAULT_FORWARD_FANOUT,
        forward_limit: int = DEFAULT_FORWARD_LIMIT,
    ) -> None:
        self._neighbours = neighbours
        self._backward_hops = backward_hops
        self._forward_fanout = forward_fanout
        self._forward_limit = forward_limit
        self.seed: dict[str, Any] | None = None

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        return self._neighbours.fetch_by_ids(ids)

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        seed = self._neighbours.resolve(query)
        if seed is None:
            raise AnchorError(
                f"could not resolve {query!r} to a paper - give an OpenAlex id "
                "(W...), a DOI, or a link containing one"
            )
        self.seed = seed
        collected: dict[str, dict[str, Any]] = {}
        seed_id = _short_id(seed.get("id"))
        collected[seed_id] = seed

        # Forward first: what built on this paper is the scarcer, more expensive
        # half, so it gets its share of the budget before backward fills the rest.
        forward_seeds = [seed_id]
        citing = self._neighbours.cited_by(seed_id, limit=self._forward_limit)
        self._absorb(collected, citing, limit)
        if self._forward_fanout > 0:
            ranked = sorted(
                (w for w in citing if _short_id(w.get("id")) != seed_id),
                key=lambda w: int(w.get("cited_by_count") or 0),
                reverse=True,
            )
            for work in ranked[: self._forward_fanout]:
                if len(collected) >= limit:
                    break
                more = self._neighbours.cited_by(
                    _short_id(work.get("id")), limit=self._forward_limit
                )
                self._absorb(collected, more, limit)
                forward_seeds.append(_short_id(work.get("id")))

        # Backward: free, because references arrive inline on works already held.
        frontier = list(collected.values())
        for _hop in range(self._backward_hops):
            if len(collected) >= limit:
                break
            wanted: list[str] = []
            for work in frontier:
                for ref in work.get("referenced_works") or []:
                    rid = _short_id(ref)
                    if rid and rid not in collected:
                        wanted.append(rid)
            if not wanted:
                break
            room = max(0, limit - len(collected))
            fetched = self._neighbours.fetch_by_ids(list(dict.fromkeys(wanted))[:room])
            frontier = self._absorb(collected, fetched, limit)
            if not frontier:
                break
        return list(collected.values())[:limit]

    @staticmethod
    def _absorb(
        collected: dict[str, dict[str, Any]],
        works: Sequence[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        added: list[dict[str, Any]] = []
        for work in works:
            if len(collected) >= limit:
                break
            wid = _short_id(work.get("id"))
            if wid and wid not in collected:
                collected[wid] = work
                added.append(work)
        return added


def anchor_label(seed: dict[str, Any] | None, reference: str) -> str:
    """Human-readable name for the graph, so the picker does not show a raw id."""
    title = str((seed or {}).get("display_name") or (seed or {}).get("title") or "").strip()
    year = (seed or {}).get("publication_year")
    if not title:
        return f"Anchored: {reference}"
    return f"Anchored: {title}" + (f" ({year})" if year else "")


def run_anchored_harvest(
    neighbours: NeighbourSource,
    reference: str,
    *,
    limit: int = 200,
    backward_hops: int = DEFAULT_BACKWARD_HOPS,
    forward_fanout: int = DEFAULT_FORWARD_FANOUT,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    expand_rounds: int = 0,
) -> HarvestResult:
    """Build a graph around one paper, reusing the whole harvest pipeline.

    The result's ``query`` is relabelled to the paper's title, since it becomes the
    graph's name everywhere downstream and a bare ``W…`` id is unreadable in a
    picker. Snowball expansion defaults **off** here: an anchored set is already
    citation-closed around its seed, so the leakage-driven expansion that helps a
    keyword harvest mostly pulls in unrelated foundations.
    """
    source: WorksSource = AnchoredSource(
        neighbours, backward_hops=backward_hops, forward_fanout=forward_fanout
    )
    result = run_harvest(
        source,
        reference,
        limit=limit,
        cache_dir=cache_dir,
        use_cache=use_cache,
        expand_rounds=expand_rounds,
    )
    seed = getattr(source, "seed", None)
    return replace(result, query=anchor_label(seed, reference))
