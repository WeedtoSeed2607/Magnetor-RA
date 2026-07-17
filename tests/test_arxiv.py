"""ArxivSource parsing and date-bounded pagination, HTTP fully mocked."""

from __future__ import annotations

import datetime as dt

import httpx
import pytest
from pytest_httpx import HTTPXMock

from magnetor import _http
from magnetor.errors import ParseError
from magnetor.sources.arxiv import ArxivSource
from magnetor.types import Domain


def _entry(arxiv_id: str, published: str) -> str:
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <title>Paper {arxiv_id}</title>
    <summary>Body.</summary>
    <published>{published}</published>
    <updated>{published}</updated>
    <author><name>Author</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}"/>
  </entry>"""


def _feed(*entries: str) -> str:
    body = "".join(entries)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">'
        f"{body}</feed>"
    )

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2606.01234v1</id>
    <title>Entangled   Widgets</title>
    <summary>We study   widgets.</summary>
    <published>2026-06-20T10:00:00Z</published>
    <updated>2026-06-21T10:00:00Z</updated>
    <author><name>Alice Q. Physicist</name></author>
    <author><name>Bob Theorist</name></author>
    <arxiv:doi>10.1000/xyz123</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/2606.01234v1"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2606.00002v1</id>
    <title>Old Result</title>
    <summary>From last week.</summary>
    <published>2026-06-01T10:00:00Z</published>
    <updated>2026-06-01T10:00:00Z</updated>
    <author><name>Carol</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2606.00002v1"/>
  </entry>
</feed>
"""


def test_parses_entry_fields(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_ATOM)
    with httpx.Client() as client:
        source = ArxivSource(Domain.QUANTUM_MECHANICS, client=client)
        papers = list(source.fetch(since=dt.datetime(2026, 6, 15, tzinfo=dt.UTC), limit=10))

    assert len(papers) == 1  # second entry is below the `since` floor
    paper = papers[0]
    assert paper.external_id == "2606.01234v1"
    assert paper.title == "Entangled Widgets"  # whitespace collapsed
    assert paper.abstract == "We study widgets."
    assert paper.authors == ("Alice Q. Physicist", "Bob Theorist")
    assert paper.doi == "10.1000/xyz123"
    assert paper.pdf_url == "http://arxiv.org/pdf/2606.01234v1"
    assert paper.full_text_available is True
    assert paper.published == dt.datetime(2026, 6, 20, 10, tzinfo=dt.UTC)


def test_stops_at_since_floor(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_ATOM)
    with httpx.Client() as client:
        source = ArxivSource(Domain.QUANTUM_MECHANICS, client=client)
        papers = list(source.fetch(since=dt.datetime(2026, 6, 25, tzinfo=dt.UTC), limit=10))
    assert papers == []


def test_rejects_unsupported_domain() -> None:
    with pytest.raises(ParseError):
        ArxivSource(Domain.NEUROSCIENCE)


def test_malformed_xml_raises_parse_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text="<not-xml")
    with httpx.Client() as client:
        source = ArxivSource(Domain.MATHEMATICS, client=client)
        with pytest.raises(ParseError):
            list(source.fetch(since=None, limit=5))


def test_pagination_stops_at_floor_across_pages(httpx_mock: HTTPXMock) -> None:
    # page_size=1 forces one entry per request; the source must page until it
    # crosses below the floor.
    httpx_mock.add_response(text=_feed(_entry("2606.10003v1", "2026-06-20T10:00:00Z")))
    httpx_mock.add_response(text=_feed(_entry("2606.10002v1", "2026-06-18T10:00:00Z")))
    httpx_mock.add_response(text=_feed(_entry("2606.10001v1", "2026-06-05T10:00:00Z")))
    with httpx.Client() as client:
        source = ArxivSource(
            Domain.QUANTUM_MECHANICS,
            client=client,
            page_size=1,
            throttle=_http.NO_THROTTLE,
        )
        papers = list(source.fetch(since=dt.datetime(2026, 6, 15, tzinfo=dt.UTC), limit=10))
    assert [p.external_id for p in papers] == ["2606.10003v1", "2606.10002v1"]
    assert len(httpx_mock.get_requests()) == 3  # third page crossed the floor


def test_pagination_stops_at_limit_across_pages(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_feed(_entry("2606.20002v1", "2026-06-20T10:00:00Z")))
    httpx_mock.add_response(text=_feed(_entry("2606.20001v1", "2026-06-19T10:00:00Z")))
    with httpx.Client() as client:
        source = ArxivSource(
            Domain.QUANTUM_MECHANICS,
            client=client,
            page_size=1,
            throttle=_http.NO_THROTTLE,
        )
        papers = list(source.fetch(since=dt.datetime(2026, 6, 1, tzinfo=dt.UTC), limit=2))
    assert len(papers) == 2
    assert len(httpx_mock.get_requests()) == 2
