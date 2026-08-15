"""PubMedCentralSource: esearch + efetch + esummary flow, HTTP fully mocked."""

from __future__ import annotations

import datetime as dt
import json
from xml.etree import ElementTree as ET

import httpx
from pytest_httpx import HTTPXMock

from magnetor import _http
from magnetor.sources.pubmed import (
    PubMedCentralSource,
    _abstract_text,
    _parse_pubdate,
)
from magnetor.types import DatePrecision, Domain

_ESEARCH = json.dumps({"esearchresult": {"idlist": ["111", "222"]}})

# efetch returns JATS; real PMC uses <article-id pub-id-type="pmcid">PMC111</...>
# (already prefixed) alongside a bare "pmcaid". Only PMC111 has an abstract here.
_EFETCH = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmcid">PMC111</article-id>
      <article-id pub-id-type="pmcaid">111</article-id>
      <abstract><title>ABSTRACT</title><p>We map   cortical dynamics in detail.</p></abstract>
    </article-meta></front>
  </article>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmcid">PMC222</article-id>
    </article-meta></front>
  </article>
</pmc-articleset>
"""

_ESUMMARY = json.dumps(
    {
        "result": {
            "uids": ["111", "222"],
            "111": {
                "title": "Cortical   Dynamics",
                "pubdate": "2026 Jun 20",
                "authors": [{"name": "N. Euron"}, {"name": "G. Lia"}],
                "articleids": [{"idtype": "doi", "value": "10.1234/neuro.1"}],
            },
            "222": {
                "title": "Synaptic Pruning",
                "pubdate": "2026 Jun",
                "authors": [{"name": "A. Xon"}],
                "articleids": [],
            },
        }
    }
)


def _source(client: httpx.Client, **kwargs: object) -> PubMedCentralSource:
    return PubMedCentralSource(client=client, throttle=_http.NO_THROTTLE, **kwargs)  # type: ignore[arg-type]


def test_search_efetch_summary(httpx_mock: HTTPXMock) -> None:
    # Responses are consumed in registration order: esearch, efetch, esummary.
    httpx_mock.add_response(text=_ESEARCH)
    httpx_mock.add_response(text=_EFETCH)
    httpx_mock.add_response(text=_ESUMMARY)

    with httpx.Client() as client:
        source = _source(client, email="test@example.com")
        papers = list(source.fetch(since=dt.datetime(2026, 6, 1, tzinfo=dt.UTC), limit=10))

    # The esearch request must carry the open-access filter so the
    # full_text_available / license flags we set are truthful.
    esearch_term = httpx_mock.get_requests()[0].url.params["term"]
    assert '"open access"[filter]' in esearch_term

    assert source.domain is Domain.NEUROSCIENCE
    assert len(papers) == 2
    first = papers[0]
    assert first.external_id == "PMC111"
    assert first.title == "Cortical Dynamics"
    assert first.authors == ("N. Euron", "G. Lia")
    assert first.doi == "10.1234/neuro.1"
    assert first.full_text_available is True
    assert first.published == dt.datetime(2026, 6, 20, tzinfo=dt.UTC)
    assert first.published_precision is DatePrecision.DAY
    # Second record's pubdate is "2026 Jun": the same datetime a day-exact
    # "2026 Jun 01" would give, so only the precision distinguishes them.
    assert papers[1].published == dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
    assert papers[1].published_precision is DatePrecision.MONTH
    # Abstract enrichment from efetch: section <title>s are included as text
    # (structured abstracts carry labels like "ABSTRACT", "Methods", ...).
    assert first.abstract == "ABSTRACT We map cortical dynamics in detail."
    # Second article had no abstract in efetch -> empty, not an error.
    assert papers[1].abstract == ""
    # dedup key prefers DOI when present.
    assert first.dedup_key() == "doi:10.1234/neuro.1"
    assert papers[1].dedup_key() == "PubMed Central:PMC222"


def _abstract(inner: str) -> ET.Element:
    return ET.fromstring(f"<abstract>{inner}</abstract>")


def test_a_structured_abstract_keeps_the_labels_the_publisher_supplied() -> None:
    """Without the colon, "Methods We recruited..." is indistinguishable from prose."""
    text = _abstract_text(
        _abstract(
            "<sec><title>Background</title><p>Knee pain is common.</p></sec>"
            "<sec><title>Methods</title><p>We recruited   40 adults.</p></sec>"
            "<sec><title>Conclusions</title><p>Exercise helped.</p></sec>"
        )
    )
    assert text == (
        "Background: Knee pain is common. Methods: We recruited 40 adults. "
        "Conclusions: Exercise helped."
    )


def test_an_unstructured_abstract_is_joined_exactly_as_before() -> None:
    """Physics and maths abstracts are one block — there is no structure to keep."""
    text = _abstract_text(_abstract("<title>ABSTRACT</title><p>We map   dynamics.</p>"))
    assert text == "ABSTRACT We map dynamics."


def test_a_section_without_a_title_keeps_its_body() -> None:
    text = _abstract_text(
        _abstract("<sec><p>No label here.</p></sec><sec><title>Aim</title><p>To test.</p></sec>")
    )
    assert text == "No label here. Aim: To test."


def test_text_outside_a_section_is_not_dropped() -> None:
    text = _abstract_text(
        _abstract("<p>Lead-in sentence.</p><sec><title>Methods</title><p>We ran it.</p></sec>")
    )
    assert text == "Lead-in sentence. Methods: We ran it."


def test_a_year_only_pubdate_records_that_it_is_year_only() -> None:
    """``strptime`` turns "2026" into January 1st. Only the precision says so."""
    parsed = _parse_pubdate("2026")
    assert parsed is not None
    stamp, precision = parsed
    assert stamp == dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    assert precision is DatePrecision.YEAR


def test_each_pubdate_shape_reports_its_own_precision() -> None:
    cases = {
        "2026 Jun 20": DatePrecision.DAY,
        "2026 Jun": DatePrecision.MONTH,
        "2026": DatePrecision.YEAR,
    }
    for value, expected in cases.items():
        parsed = _parse_pubdate(value)
        assert parsed is not None, value
        assert parsed[1] is expected, value


def test_an_unparseable_pubdate_yields_no_date_at_all() -> None:
    assert _parse_pubdate("") is None
    assert _parse_pubdate("not a date") is None


def test_empty_search_returns_no_papers(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=json.dumps({"esearchresult": {"idlist": []}}))
    with httpx.Client() as client:
        source = _source(client)
        assert list(source.fetch(since=None, limit=10)) == []
