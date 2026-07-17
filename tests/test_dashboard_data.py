"""Dashboard view-model helpers (Spec 11): banner text, frontier feed, trend load.

These cover the pure, Streamlit-free layer. The Streamlit UI (dashboard.py) is a
thin render shell and is deliberately not imported here.
"""

from __future__ import annotations

import datetime as dt
import json

from magnetor.citations import Citation
from magnetor.config import get_domain_config
from magnetor.dashboard_data import (
    FrontierFeed,
    banner_lines,
    citation_url,
    frontier_feed,
    linked,
    load_trends,
    paper_url,
    search_access,
)
from magnetor.resources import DomainStore
from magnetor.trends import TRENDS_FILENAME
from magnetor.types import Domain, Paper
from tests.conftest import make_paper

_QM = Domain.QUANTUM_MECHANICS


def _store() -> DomainStore:
    return DomainStore(_QM, get_domain_config(_QM).storage_dir)


def _write_trends(payload: dict[str, object]) -> None:
    path = get_domain_config(_QM).storage_dir / TRENDS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_trends_returns_none_when_absent() -> None:
    assert load_trends(_QM) is None


def test_load_trends_reads_persisted_json() -> None:
    _write_trends({"domain": "qm", "interpretation": ["x"]})
    loaded = load_trends(_QM)
    assert loaded is not None
    assert loaded["domain"] == "qm"


def test_banner_lines_from_interpretation() -> None:
    trends: dict[str, object] = {
        "interpretation": ["Topic 0 rose 12 points.", "Emerging term 'foo'."]
    }
    assert banner_lines(trends) == [
        "Topic 0 rose 12 points.",
        "Emerging term 'foo'.",
    ]


def test_banner_lines_empty_without_trends() -> None:
    assert banner_lines(None) == []
    assert banner_lines({"topics": []}) == []  # no interpretation key


def test_frontier_feed_orders_hot_clusters_by_latest_prevalence() -> None:
    _write_trends(
        {
            "anomalies": [{"term": "transformer", "delta": 0.9}],
            "topics": [
                {"id": 0, "keywords": ["a"], "prevalence": [0.1, 0.2]},
                {"id": 1, "keywords": ["b"], "prevalence": [0.1, 0.8]},
            ],
        }
    )
    store = _store()
    store.store(make_paper(external_id="p1", published=dt.datetime(2026, 6, 1, tzinfo=dt.UTC)))

    feed = frontier_feed(_QM, load_trends(_QM), recent=5)

    assert isinstance(feed, FrontierFeed)
    # Topic 1 has the higher latest-slice prevalence -> first.
    assert [t["id"] for t in feed.hot_clusters] == [1, 0]
    assert feed.anomalies[0]["term"] == "transformer"
    assert len(feed.recent) == 1


def test_frontier_feed_without_trends_still_lists_recent() -> None:
    store = _store()
    for i in range(3):
        store.store(make_paper(external_id=f"p{i}"))
    feed = frontier_feed(_QM, None, recent=2)
    assert feed.anomalies == ()
    assert feed.hot_clusters == ()
    assert len(feed.recent) == 2


def _paper(**kw: object) -> Paper:
    base: dict[str, object] = dict(
        domain=_QM, source="test", external_id="x", title="T",
        abstract="a", authors=("A",), published=None,
    )
    base.update(kw)
    return Paper(**base)  # type: ignore[arg-type]


def test_paper_url_arxiv_uses_abstract_page() -> None:
    p = _paper(source="arXiv", external_id="2607.08462v1", pdf_url="https://x/pdf")
    assert paper_url(p) == "https://arxiv.org/abs/2607.08462v1"


def test_paper_url_pubmed_uses_pmc_article() -> None:
    p = _paper(source="PubMed Central", external_id="PMC13373730", doi="10.1/x")
    assert paper_url(p) == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13373730/"


def test_paper_url_falls_back_to_doi_then_pdf() -> None:
    assert paper_url(_paper(source="other", external_id="z", doi="10.5/y")) == "https://doi.org/10.5/y"
    assert paper_url(_paper(source="other", external_id="z", pdf_url="https://p")) == "https://p"
    assert paper_url(_paper(source="other", external_id="z")) is None


def test_citation_url_prefers_doi_then_arxiv() -> None:
    assert citation_url(Citation(title="t", year=None, doi="10.9/z", arxiv_id=None)) == "https://doi.org/10.9/z"
    assert citation_url(Citation(title="t", year=None, doi=None, arxiv_id="2601.1")) == "https://arxiv.org/abs/2601.1"
    assert citation_url(Citation(title="t", year=None, doi=None, arxiv_id=None)) is None


def test_linked_emits_markdown_only_with_url() -> None:
    assert linked("Title", "https://u") == "[Title](https://u)"
    assert linked("Title", None) == "Title"


def test_search_access_open_when_no_password_configured() -> None:
    # Local dev: no password set -> always unlocked.
    assert search_access(None, None, False) == (True, None)
    assert search_access("", "anything", False) == (True, None)


def test_search_access_gates_and_validates_password() -> None:
    # Wrong / no entry -> locked; correct entry -> unlocked.
    assert search_access("s3cret", None, False) == (False, None)
    assert search_access("s3cret", "nope", False) == (False, "Incorrect password")
    assert search_access("s3cret", "s3cret", False) == (True, None)


def test_search_access_stays_unlocked_once_ok() -> None:
    # Already unlocked this session -> stays unlocked without re-entry.
    assert search_access("s3cret", None, True) == (True, None)
