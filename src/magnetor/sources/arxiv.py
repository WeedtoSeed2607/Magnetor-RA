"""arXiv acquisition source (Quantum Mechanics and Mathematics).

Uses the public arXiv Atom API. One instance is bound to one domain via its
category filter, so a single class serves both QM (``quant-ph``) and Math
(``math.*``) without crossing the domain boundary.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from xml.etree import ElementTree as ET

import httpx

from magnetor import _http
from magnetor.errors import ParseError
from magnetor.types import Domain, Paper

_API_URL = "https://export.arxiv.org/api/query"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv category per domain. Only the two arXiv-backed domains are mapped.
# NOTE: math papers are subcategorised (math.AG, math.NT, ...), so the bare
# "math" archive matches nothing — the "math.*" wildcard is required. quant-ph
# has no subcategories, so it is used as-is.
_CATEGORY: dict[Domain, str] = {
    Domain.QUANTUM_MECHANICS: "quant-ph",
    Domain.MATHEMATICS: "math.*",
}

#: Cold-start lookback when there is no prior run to bound from. Deliberately a
#: week, not a day: the first run must seed the store across normal submission
#: gaps (weekends/holidays). Steady-state runs are bounded by the last run.
_COLD_START = dt.timedelta(days=7)

#: arXiv's usage terms ask for ~3 seconds between programmatic requests.
_REQUEST_INTERVAL = 3.0


class ArxivSource:
    """Fetch recent papers for one arXiv-backed domain."""

    def __init__(
        self,
        domain: Domain,
        *,
        client: httpx.Client | None = None,
        page_size: int = 100,
        throttle: _http.Throttle | None = None,
    ) -> None:
        if domain not in _CATEGORY:
            raise ParseError(f"ArxivSource does not serve domain {domain!r}")
        self._domain = domain
        self._category = _CATEGORY[domain]
        self._client = client
        self._page_size = page_size
        self._throttle = throttle or _http.Throttle(_REQUEST_INTERVAL)

    @property
    def domain(self) -> Domain:
        return self._domain

    def fetch(self, *, since: dt.datetime | None, limit: int) -> Iterable[Paper]:
        """Yield papers submitted at/after ``since``, newest first, up to ``limit``.

        arXiv sorts by submission date descending; we stop as soon as we cross
        below ``since`` so we never page through the whole archive.
        """
        floor = since or (_utcnow() - _COLD_START)
        yielded = 0
        start = 0
        while yielded < limit:
            want = min(self._page_size, limit - yielded)
            xml = _http.get(
                _API_URL,
                params={
                    "search_query": f"cat:{self._category}",
                    "start": str(start),
                    "max_results": str(want),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                client=self._client,
                throttle=self._throttle,
            )
            page = list(self._parse(xml))
            if not page:
                break
            for paper in page:
                if paper.published is not None and paper.published < floor:
                    return
                yield paper
                yielded += 1
                if yielded >= limit:
                    return
            start += len(page)

    def _parse(self, xml: str) -> Iterator[Paper]:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            raise ParseError(f"arXiv returned unparseable XML: {exc}") from exc
        for entry in root.findall("atom:entry", _NS):
            yield self._entry_to_paper(entry)

    def _entry_to_paper(self, entry: ET.Element) -> Paper:
        raw_id = _text(entry.find("atom:id", _NS))
        external_id = raw_id.rsplit("/abs/", 1)[-1] if raw_id else ""
        authors = tuple(
            _text(name)
            for name in entry.findall("atom:author/atom:name", _NS)
            if _text(name)
        )
        pdf_url = None
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        doi = _text(entry.find("arxiv:doi", _NS)) or None
        return Paper(
            domain=self._domain,
            source="arXiv",
            external_id=external_id,
            title=_clean(_text(entry.find("atom:title", _NS))),
            abstract=_clean(_text(entry.find("atom:summary", _NS))),
            authors=authors,
            published=_parse_dt(_text(entry.find("atom:published", _NS))),
            updated=_parse_dt(_text(entry.find("atom:updated", _NS))),
            doi=doi,
            pdf_url=pdf_url,
            full_text_available=pdf_url is not None,
            license="arXiv (open access, full text)",
        )


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def _text(element: ET.Element | None) -> str:
    return element.text or "" if element is not None else ""


def _clean(value: str) -> str:
    return " ".join(value.split())


def _parse_dt(value: str) -> dt.datetime | None:
    if not value:
        return None
    # arXiv uses e.g. "2026-06-20T17:30:00Z"; normalise the trailing Z.
    normalised = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalised)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
