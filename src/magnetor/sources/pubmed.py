"""PubMed Central acquisition source (Neuroscience).

Uses NCBI E-utilities: ``esearch`` to find recent PMC ids for a neuroscience
query, ``esummary`` for lightweight metadata, and ``efetch`` to enrich each
record with its abstract (esummary omits abstracts, but the trend and matching
layers need them). Full-text bodies live in the PMC open-access subset and are
recorded as a pointer only; bulk full-text retrieval is a later phase.

NCBI etiquette: a ``tool`` name and, if configured, an ``email`` and API key are
sent so requests are identifiable; a throttle spaces requests under the rate
limit (3 req/s without a key, 10 req/s with one).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable
from xml.etree import ElementTree as ET

import httpx

from magnetor import _http
from magnetor.errors import ParseError
from magnetor.types import DatePrecision, Domain, Paper

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_TOOL = "magnetor"
_DEFAULT_QUERY = "neuroscience"
#: Restrict results to the PMC open-access subset so the full-text/license flags
#: we set are truthful and abstracts/bodies are actually retrievable. Without it,
#: db=pmc also returns non-OA records we would mislabel as open access.
_OA_FILTER = '"open access"[filter]'
#: Week-long first-run lookback to seed the store; steady state is bounded by
#: the last run (matches the arXiv source's cold-start rationale).
_COLD_START = dt.timedelta(days=7)

#: Min seconds between requests: 3 req/s without a key, 10 req/s with one.
_INTERVAL_NO_KEY = 1.0 / 3
_INTERVAL_WITH_KEY = 1.0 / 10


class PubMedCentralSource:
    """Fetch recent open-access neuroscience papers from PubMed Central."""

    domain = Domain.NEUROSCIENCE

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        query: str = _DEFAULT_QUERY,
        email: str | None = None,
        api_key: str | None = None,
        throttle: _http.Throttle | None = None,
    ) -> None:
        self._client = client
        self._query = query
        # Contact details are configuration, never hard-coded: read from env so
        # the same code runs politely in any deployment.
        self._email = email or os.environ.get("MAGNETOR_NCBI_EMAIL")
        self._api_key = api_key or os.environ.get("MAGNETOR_NCBI_API_KEY")
        interval = _INTERVAL_WITH_KEY if self._api_key else _INTERVAL_NO_KEY
        self._throttle = throttle or _http.Throttle(interval)

    def _common_params(self) -> dict[str, str]:
        params = {"db": "pmc", "tool": _TOOL}
        if self._email:
            params["email"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def fetch(self, *, since: dt.datetime | None, limit: int) -> Iterable[Paper]:
        """Yield up to ``limit`` neuroscience papers published at/after ``since``."""
        floor = since or (_utcnow() - _COLD_START)
        ids = self._search(floor=floor, limit=limit)
        if not ids:
            return []
        abstracts = self._abstracts(ids)
        return self._summaries(ids, abstracts)

    def _get(self, url: str, params: dict[str, str]) -> str:
        return _http.get(url, params=params, client=self._client, throttle=self._throttle)

    def _search(self, *, floor: dt.datetime, limit: int) -> list[str]:
        params = self._common_params() | {
            "term": f"({self._query}) AND {_OA_FILTER}",
            "retmax": str(limit),
            "retmode": "json",
            "sort": "pub_date",
            "datetype": "pdat",
            "mindate": floor.strftime("%Y/%m/%d"),
            "maxdate": _utcnow().strftime("%Y/%m/%d"),
        }
        payload = _load_json(self._get(_ESEARCH_URL, params))
        block = _expect_dict(payload.get("esearchresult"), "esearchresult")
        idlist = block.get("idlist")
        if not isinstance(idlist, list):
            raise ParseError("esearch payload missing an idlist")
        return [str(uid) for uid in idlist]

    def _abstracts(self, ids: list[str]) -> dict[str, str]:
        """Fetch abstracts via efetch, keyed by ``PMC<id>``.

        Enrichment: a paper with no abstract is still stored, just without one.
        """
        params = self._common_params() | {
            "id": ",".join(ids),
            "retmode": "xml",
        }
        return _parse_abstracts(self._get(_EFETCH_URL, params))

    def _summaries(self, ids: list[str], abstracts: dict[str, str]) -> list[Paper]:
        params = self._common_params() | {
            "id": ",".join(ids),
            "retmode": "json",
        }
        payload = _load_json(self._get(_ESUMMARY_URL, params))
        result = _expect_dict(payload.get("result"), "result")
        uids = result.get("uids", [])
        if not isinstance(uids, list):
            raise ParseError("esummary payload missing a uids list")
        papers: list[Paper] = []
        for raw_uid in uids:
            uid = str(raw_uid)
            record = result.get(uid)
            if isinstance(record, dict):
                papers.append(self._record_to_paper(uid, record, abstracts))
        return papers

    def _record_to_paper(
        self, uid: str, record: dict[str, object], abstracts: dict[str, str]
    ) -> Paper:
        authors = tuple(
            str(a["name"])
            for a in _as_list(record.get("authors"))
            if isinstance(a, dict) and "name" in a
        )
        doi = None
        for aid in _as_list(record.get("articleids")):
            if isinstance(aid, dict) and aid.get("idtype") == "doi":
                doi = str(aid.get("value")) or None
                break
        external_id = f"PMC{uid}"
        dated = _parse_pubdate(str(record.get("pubdate", "")))
        return Paper(
            domain=self.domain,
            source="PubMed Central",
            external_id=external_id,
            title=_clean(str(record.get("title", ""))),
            abstract=abstracts.get(external_id, ""),
            authors=authors,
            published=dated[0] if dated else None,
            published_precision=dated[1] if dated else None,
            doi=doi,
            pdf_url=None,
            # Search is OA-filtered (_OA_FILTER), so these flags are truthful;
            # bulk full-text retrieval itself is deferred to a later phase.
            full_text_available=True,
            license="PMC open-access subset",
        )


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def _load_json(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"NCBI returned non-JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError("NCBI JSON payload was not an object")
    return payload


def _expect_dict(value: object, label: str) -> dict[str, object]:
    """Narrow an untyped JSON value to a dict or fail at the boundary."""
    if not isinstance(value, dict):
        raise ParseError(f"NCBI payload missing expected object {label!r}")
    return value


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


#: article-id types carrying the PMC accession, in preference order. Real PMC
#: efetch emits ``pmcid`` (value already ``PMC123``) plus ``pmcaid`` (bare
#: ``123``); we normalise both to a ``PMC<digits>`` key.
_PMC_ID_TYPES = ("pmcid", "pmc", "pmcaid", "pmcaiid")


def _parse_abstracts(xml: str) -> dict[str, str]:
    """Map ``PMC<id>`` -> abstract text from an efetch JATS article set.

    Each ``<article>`` carries its PMC accession in an ``<article-id>`` and its
    abstract under ``<abstract>``. Articles without an abstract are skipped.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ParseError(f"efetch returned unparseable XML: {exc}") from exc
    result: dict[str, str] = {}
    for article in root.iter("article"):
        pmc_id = _article_pmc_id(article)
        if pmc_id is None:
            continue
        abstract = next(article.iter("abstract"), None)
        if abstract is not None:
            text = _abstract_text(abstract)
            if text:
                result[pmc_id] = text
    return result


def _abstract_text(abstract: ET.Element) -> str:
    """Flatten a JATS abstract, keeping any section labels the publisher supplied.

    Structured abstracts — the medical convention since the Ad Hoc Working Group
    proposed them in 1987 — wrap each part as
    ``<sec><title>Methods</title><p>...</p></sec>``. Joining every string with
    ``itertext()`` ran the label straight into the sentence after it,
    ``"Methods We recruited..."``, which reads as ordinary prose and cannot be
    told apart from it afterwards. Writing ``"Methods: We recruited..."`` keeps a
    structure a reader, and a later parser, can still see.

    Abstracts with no ``<sec>`` are joined exactly as before: there is no
    structure to keep, and most physics and maths abstracts are a single block.
    """
    if not abstract.findall("sec"):
        return _clean(" ".join(abstract.itertext()))
    parts: list[str] = []
    for child in abstract:
        if child.tag != "sec":
            loose = _clean(" ".join(child.itertext()))
            if loose:
                parts.append(loose)
            continue
        title_element = child.find("title")
        label = (
            _clean(" ".join(title_element.itertext()))
            if title_element is not None
            else ""
        )
        body = _clean(
            " ".join(
                text
                for element in child
                if element is not title_element
                for text in element.itertext()
            )
        )
        section = f"{label}: {body}" if label and body else (body or label)
        if section:
            parts.append(section)
    return _clean(" ".join(parts))


def _article_pmc_id(article: ET.Element) -> str | None:
    """Extract and normalise the ``PMC<digits>`` accession from an article."""
    for aid in article.iter("article-id"):
        if aid.get("pub-id-type") in _PMC_ID_TYPES:
            raw = (aid.text or "").strip()
            if raw:
                return raw if raw.upper().startswith("PMC") else f"PMC{raw}"
    return None


def _clean(value: str) -> str:
    return " ".join(value.split())


#: Accepted ``pubdate`` shapes, most precise first, each with what it actually
#: pins down. Order matters: ``%Y`` would happily match the year off a fuller
#: string if tried first.
_PUBDATE_FORMATS: tuple[tuple[str, DatePrecision], ...] = (
    ("%Y %b %d", DatePrecision.DAY),
    ("%Y %b", DatePrecision.MONTH),
    ("%Y", DatePrecision.YEAR),
)


def _parse_pubdate(value: str) -> tuple[dt.datetime, DatePrecision] | None:
    """Parse an E-utilities pubdate, keeping how precise it actually was.

    ``strptime`` fills absent components with January 1st, so ``2026`` and
    ``2026 Jan 01`` produce the identical datetime. Returning the precision
    alongside is the only thing that lets a caller tell a year-wide record from
    a day-exact one — see :mod:`magnetor.dating`.
    """
    if not value:
        return None
    for fmt, precision in _PUBDATE_FORMATS:
        try:
            parsed = dt.datetime.strptime(value, fmt).replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        return parsed, precision
    return None
