"""SemanticScholarClient: id mapping, parsing, graceful degradation."""

from __future__ import annotations

import json

import httpx
from pytest_httpx import HTTPXMock

from magnetor import _http
from magnetor.citations import SemanticScholarClient, _paper_id
from magnetor.types import Domain
from tests.conftest import make_paper


def _client(client: httpx.Client) -> SemanticScholarClient:
    return SemanticScholarClient(client=client, throttle=_http.NO_THROTTLE)


def _citations_body(*titles: str) -> str:
    return json.dumps(
        {"data": [{"citingPaper": {"title": t, "year": 2026, "externalIds": {}}} for t in titles]}
    )


def _references_body(*titles: str) -> str:
    return json.dumps(
        {
            "data": [
                {"citedPaper": {"title": t, "year": 2025, "externalIds": {"DOI": "10.1/x"}}}
                for t in titles
            ]
        }
    )


def test_paper_id_prefers_doi() -> None:
    assert _paper_id(make_paper(doi="10.1/abc")) == "DOI:10.1/abc"


def test_paper_id_arxiv_strips_version() -> None:
    paper = make_paper(external_id="2606.01234v3")
    object.__setattr__(paper, "source", "arXiv")
    assert _paper_id(paper) == "ARXIV:2606.01234"


def test_paper_id_pmc() -> None:
    paper = make_paper(external_id="PMC13340929")
    object.__setattr__(paper, "source", "PubMed Central")
    assert _paper_id(paper) == "PMCID:PMC13340929"


def test_paper_id_none_when_unresolvable() -> None:
    paper = make_paper(external_id="local-1")
    object.__setattr__(paper, "source", "manual")
    assert _paper_id(paper) is None


def test_expand_returns_forward_and_backward(httpx_mock: HTTPXMock) -> None:
    # First call = citations (forward), second = references (backward).
    httpx_mock.add_response(text=_citations_body("Citing A", "Citing B"))
    httpx_mock.add_response(text=_references_body("Cited X"))
    with httpx.Client() as client:
        forward, backward = _client(client).expand(make_paper(doi="10.1/abc"))
    assert [c.title for c in forward] == ["Citing A", "Citing B"]
    assert [c.title for c in backward] == ["Cited X"]
    assert backward[0].doi == "10.1/x"


def test_no_resolvable_id_skips_network(httpx_mock: HTTPXMock) -> None:
    paper = make_paper(external_id="x")
    object.__setattr__(paper, "source", "manual")
    with httpx.Client() as client:
        assert _client(client).expand(paper) == ([], [])
    assert httpx_mock.get_requests() == []


def test_degrades_to_empty_on_error(httpx_mock: HTTPXMock) -> None:
    # 404 / rate-limit -> SourceUnavailableError -> empty, not a crash.
    httpx_mock.add_response(status_code=404, is_reusable=True)
    with httpx.Client() as client:
        forward, backward = SemanticScholarClient(
            client=client, throttle=_http.NO_THROTTLE
        ).expand(make_paper(doi="10.1/missing"))
    assert forward == [] and backward == []


def test_api_key_sent_as_header(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_citations_body("A"))
    httpx_mock.add_response(text=_references_body("B"))
    with httpx.Client() as client:
        SemanticScholarClient(
            api_key="s2-key", client=client, throttle=_http.NO_THROTTLE
        ).expand(make_paper(domain=Domain.QUANTUM_MECHANICS, doi="10.1/abc"))
    assert httpx_mock.get_requests()[0].headers["x-api-key"] == "s2-key"
