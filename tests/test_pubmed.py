"""PubMedCentralSource: esearch + efetch + esummary flow, HTTP fully mocked."""

from __future__ import annotations

import datetime as dt
import json

import httpx
from pytest_httpx import HTTPXMock

from magnetor import _http
from magnetor.sources.pubmed import PubMedCentralSource
from magnetor.types import Domain

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
    # Abstract enrichment from efetch: section <title>s are included as text
    # (structured abstracts carry labels like "ABSTRACT", "Methods", ...).
    assert first.abstract == "ABSTRACT We map cortical dynamics in detail."
    # Second article had no abstract in efetch -> empty, not an error.
    assert papers[1].abstract == ""
    # dedup key prefers DOI when present.
    assert first.dedup_key() == "doi:10.1234/neuro.1"
    assert papers[1].dedup_key() == "PubMed Central:PMC222"


def test_empty_search_returns_no_papers(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=json.dumps({"esearchresult": {"idlist": []}}))
    with httpx.Client() as client:
        source = _source(client)
        assert list(source.fetch(since=None, limit=10)) == []
