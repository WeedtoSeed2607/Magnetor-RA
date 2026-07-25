"""Branch C · L1 — Topic Harvest from OpenAlex (see ADR-0006).

Harvests a query's paper set and the citation edges *within* that set — the
substrate the Evidence Graph scores over. Per ADR-0006 D1 this is an **offline
batch job**: a context-complete harvest runs for minutes and must never sit
inside a dashboard request cycle. Raw responses are cached so re-harvests never
re-fetch.

OpenAlex is free and needs no key; we join the "polite pool" by sending a
``mailto``. The fields we depend on — ``referenced_works`` (citation edges),
``is_retracted`` (integrity flag), ``type`` (review detection), and author
``institutions`` country codes (venue reach) — all arrive in one call.

Isolation note: Branch C builds a per-query working set sourced from OpenAlex; it
neither reads nor writes the per-domain silos, so the storage isolation invariant
(no body/embedding crossing a domain directory) is untouched.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from magnetor import _http
from magnetor.config import global_store_path

OPENALEX_BASE = "https://api.openalex.org/works"
DEFAULT_LIMIT = 200
_MAX_PER_PAGE = 200
_ID_BATCH = 50  # OpenAlex OR-filter caps at 50 ids per request
HARVEST_CACHE_DIR = "_harvest_cache"
WORKS_CACHE_FILE = "works.json"  # id -> raw work, so expansion re-runs stay offline
DEFAULT_EXPAND_PER_ROUND = 60  # missing foundational works pulled in per round
#: Stop expanding when a round drops boundary leakage by less than this
#: (diminishing returns — the graph is about as complete as this harvest gets).
_LEAK_PLATEAU = 0.02
#: OpenAlex payload trim — only the fields the Evidence Graph consumes.
_SELECT = (
    "id,display_name,title,publication_year,doi,cited_by_count,"
    "referenced_works,type,is_retracted,authorships,primary_location,"
    "abstract_inverted_index"
)


@dataclass(frozen=True, slots=True)
class HarvestedPaper:
    """One OpenAlex work, trimmed to the signals Branch C scores over."""

    openalex_id: str  # short form, e.g. "W4405183524"
    title: str
    year: int | None
    doi: str | None  # bare DOI, prefix stripped
    venue: str | None
    #: GLOBAL citation count — a credibility signal ONLY, never the influence
    #: metric (ADR-0006 L3.0: influence is in-subgraph, query-relative).
    cited_by_count: int
    referenced_works: tuple[str, ...]  # short OpenAlex ids this paper cites
    institution_countries: tuple[str, ...]  # ISO codes, deduped, for venue reach
    is_review: bool  # review-article endorsement signal (L3.0)
    is_retracted: bool  # integrity flag (§4.2) — free from OpenAlex
    abstract: str = ""


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """A harvested working set plus its in-subgraph citation edges."""

    query: str
    generated_at: str
    papers: tuple[HarvestedPaper, ...]
    edges: tuple[tuple[str, str], ...]  # (citing_id, cited_id); both in-set
    n_fetched: int
    #: IDs of papers whose OpenAlex referenced_works listed themselves — an
    #: upstream data error. The self-loop is dropped; the ID is surfaced so a
    #: recurrence is never silent.
    self_referencing_ids: tuple[str, ...] = ()
    #: Snowball expansion rounds run (0 = seed only), and the seed-only boundary
    #: leakage before expansion — so the CLI can report before -> after.
    expansion_rounds: int = 0
    seed_leakage: float = 0.0


class WorksSource(Protocol):
    """Boundary for the works provider — OpenAlex in production, fake in tests."""

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        ...

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        """Fetch specific works by OpenAlex id — the substrate for snowball expansion."""
        ...


class OpenAlexClient:
    """OpenAlex works client (free, no key; polite pool via ``mailto``)."""

    def __init__(
        self, *, mailto: str | None = None, throttle: _http.Throttle | None = None
    ) -> None:
        self._mailto = mailto or os.environ.get(
            "MAGNETOR_OPENALEX_MAILTO", "magnetor@example.com"
        )
        # OpenAlex polite pool allows ~10 req/s; 0.1s spacing stays well under.
        self._throttle = throttle if throttle is not None else _http.Throttle(0.1)

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        """Cursor-paginate OpenAlex search until ``limit`` works are collected."""
        results: list[dict[str, Any]] = []
        cursor: str | None = "*"
        while len(results) < limit and cursor:
            params = {
                "search": query,
                "per-page": str(min(_MAX_PER_PAGE, limit - len(results))),
                "cursor": cursor,
                "mailto": self._mailto,
                "select": _SELECT,
            }
            data = json.loads(_http.get(OPENALEX_BASE, params=params, throttle=self._throttle))
            batch = data.get("results") or []
            results.extend(batch)
            if not batch:
                break
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            cursor = next_cursor if isinstance(next_cursor, str) else None
        return results[:limit]

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        """Batch-fetch works by OpenAlex id (OR-filter, 50 per request)."""
        unique = [i for i in dict.fromkeys(ids) if i]
        results: list[dict[str, Any]] = []
        for start in range(0, len(unique), _ID_BATCH):
            batch = unique[start : start + _ID_BATCH]
            params = {
                "filter": "ids.openalex:" + "|".join(batch),
                "per-page": str(_ID_BATCH),
                "mailto": self._mailto,
                "select": _SELECT,
            }
            data = json.loads(_http.get(OPENALEX_BASE, params=params, throttle=self._throttle))
            results.extend(data.get("results") or [])
        return results


def run_harvest(
    source: WorksSource,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    now: dt.datetime | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    expand_rounds: int = 0,
    expand_per_round: int = DEFAULT_EXPAND_PER_ROUND,
) -> HarvestResult:
    """Harvest ``query``, optionally snowball-expand, and build in-set edges.

    ``expand_rounds`` > 0 pulls the most-referenced *external* works into the set
    (the foundational papers the harvest is missing), which shrinks boundary
    leakage and reaches toward the roots. Stops early once a round buys less than
    ``_LEAK_PLATEAU`` leakage reduction.
    """
    moment = now or dt.datetime.now(tz=dt.UTC)
    raw = _load_cache(query, limit, cache_dir) if use_cache else None
    if raw is None:
        raw = source.search(query, limit=limit)
        if use_cache:
            _save_cache(query, limit, raw, cache_dir)

    works_by_id: dict[str, dict[str, Any]] = {}
    for work in raw:
        wid = _short_id(work.get("id"))
        if wid:
            works_by_id[wid] = work
    seed_leakage = _leakage(works_by_id)

    rounds = 0
    if expand_rounds > 0:
        rounds = _expand(source, works_by_id, expand_rounds, expand_per_round, cache_dir, use_cache)

    papers = tuple(_parse_work(w) for w in works_by_id.values())
    edges, self_refs = _in_subgraph_edges(papers)
    return HarvestResult(
        query=query,
        generated_at=moment.isoformat(),
        papers=papers,
        edges=edges,
        n_fetched=len(works_by_id),
        self_referencing_ids=self_refs,
        expansion_rounds=rounds,
        seed_leakage=round(seed_leakage, 4),
    )


def _leakage(works_by_id: dict[str, dict[str, Any]]) -> float:
    """Fraction of references (over raw works) that point outside the current set."""
    ids = set(works_by_id)
    total = external = 0
    for work in works_by_id.values():
        for ref in work.get("referenced_works") or []:
            rid = _short_id(ref)
            if not rid:
                continue
            total += 1
            if rid not in ids:
                external += 1
    return external / total if total else 0.0


def _expand(
    source: WorksSource,
    works_by_id: dict[str, dict[str, Any]],
    max_rounds: int,
    per_round: int,
    cache_dir: Path | None,
    use_cache: bool,
) -> int:
    """Backward snowball: each round pull in the top external-in-degree works."""
    works_cache = _load_works_cache(cache_dir) if use_cache else {}
    prev_leak = _leakage(works_by_id)
    rounds = 0
    for _ in range(max_rounds):
        in_set = set(works_by_id)
        external: Counter[str] = Counter()
        for work in works_by_id.values():
            for ref in work.get("referenced_works") or []:
                rid = _short_id(ref)
                if rid and rid not in in_set:
                    external[rid] += 1
        if not external:
            break

        wanted = [wid for wid, _ in external.most_common(per_round)]
        missing = [wid for wid in wanted if wid not in works_cache]
        if missing:
            for fetched in source.fetch_by_ids(missing):
                fid = _short_id(fetched.get("id"))
                if fid:
                    works_cache[fid] = fetched

        added = 0
        for wid in wanted:
            if wid not in works_by_id and wid in works_cache:
                works_by_id[wid] = works_cache[wid]
                added += 1
        rounds += 1
        if added == 0:
            break

        curr_leak = _leakage(works_by_id)
        if prev_leak - curr_leak < _LEAK_PLATEAU:  # diminishing returns
            break
        prev_leak = curr_leak

    if use_cache:
        _save_works_cache(works_cache, cache_dir)
    return rounds


def _in_subgraph_edges(
    papers: Sequence[HarvestedPaper],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """In-set citation edges (citing -> cited), plus IDs that cited themselves.

    Self-loops (a paper's referenced_works listing its own ID — an upstream
    OpenAlex data error) are never legitimate influence edges: they inflate
    in-degree and PageRank. They are dropped here and their IDs returned so a
    recurrence is flagged rather than silently absorbed.
    """
    ids = {p.openalex_id for p in papers}
    edges: list[tuple[str, str]] = []
    self_refs: list[str] = []
    for paper in papers:
        for ref in paper.referenced_works:
            if ref == paper.openalex_id:
                self_refs.append(paper.openalex_id)
            elif ref in ids:
                edges.append((paper.openalex_id, ref))
    return tuple(edges), tuple(dict.fromkeys(self_refs))  # dedupe, keep order


def _short_id(url: str | None) -> str:
    return url.rsplit("/", 1)[-1] if url else ""


def _reconstruct_abstract(inverted: object) -> str:
    """Rebuild plain text from OpenAlex's ``abstract_inverted_index``."""
    if not isinstance(inverted, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if isinstance(idxs, list):
            positioned.extend((int(i), str(word)) for i in idxs if isinstance(i, int))
    positioned.sort()
    return " ".join(word for _, word in positioned)


def _parse_work(work: dict[str, Any]) -> HarvestedPaper:
    doi_raw = work.get("doi")
    doi = doi_raw.replace("https://doi.org/", "") or None if isinstance(doi_raw, str) else None
    source = work.get("primary_location") or {}
    venue = ((source.get("source") or {}) if isinstance(source, dict) else {}).get("display_name")

    countries: list[str] = []
    for authorship in work.get("authorships") or []:
        for inst in authorship.get("institutions") or []:
            code = inst.get("country_code")
            if isinstance(code, str):
                countries.append(code)

    year = work.get("publication_year")
    cited = work.get("cited_by_count")
    return HarvestedPaper(
        openalex_id=_short_id(work.get("id")),
        title=str(work.get("title") or work.get("display_name") or ""),
        year=year if isinstance(year, int) else None,
        doi=doi,
        venue=venue if isinstance(venue, str) else None,
        cited_by_count=cited if isinstance(cited, int) else 0,
        referenced_works=tuple(_short_id(r) for r in work.get("referenced_works") or []),
        institution_countries=tuple(dict.fromkeys(countries)),  # dedupe, keep order
        is_review=str(work.get("type") or "").lower() == "review",
        is_retracted=bool(work.get("is_retracted")),
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
    )


def _cache_path(query: str, limit: int, cache_dir: Path | None) -> Path:
    root = cache_dir or global_store_path(HARVEST_CACHE_DIR)
    key = hashlib.sha256(f"{query}|{limit}".encode()).hexdigest()[:16]
    return root / f"{key}.json"


def _load_cache(query: str, limit: int, cache_dir: Path | None) -> list[dict[str, Any]] | None:
    path = _cache_path(query, limit, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def _save_cache(query: str, limit: int, raw: list[dict[str, Any]], cache_dir: Path | None) -> None:
    path = _cache_path(query, limit, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _works_cache_path(cache_dir: Path | None) -> Path:
    root = cache_dir or global_store_path(HARVEST_CACHE_DIR)
    return root / WORKS_CACHE_FILE


def _load_works_cache(cache_dir: Path | None) -> dict[str, dict[str, Any]]:
    path = _works_cache_path(cache_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_works_cache(cache: dict[str, dict[str, Any]], cache_dir: Path | None) -> None:
    path = _works_cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
