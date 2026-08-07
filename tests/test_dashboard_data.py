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


def _graph_doc() -> dict[str, object]:
    return {
        "query": "q",
        "nodes": [
            {"id": "A", "title": "Alpha paper on codes", "influence": 1.0, "stable": True},
            {"id": "B", "title": "Beta", "influence": 0.5, "is_retracted": True},
            {"id": "C", "title": "Gamma", "influence": 0.2, "stable": False},
        ],
        "edges": [["B", "A"], ["C", "A"]],
    }


def test_graph_dot_is_a_digraph_with_nodes_and_edges() -> None:
    from magnetor.dashboard_data import graph_dot

    dot = graph_dot(_graph_doc())
    assert dot.startswith("digraph EvidenceGraph {")
    assert '"A"' in dot and '"B" -> "A"' in dot
    assert "#d9534f" in dot  # retracted node coloured red
    assert "#f0ad4e" in dot  # unstable-rank node coloured orange


def test_graph_dot_respects_top_n_and_prunes_edges() -> None:
    from magnetor.dashboard_data import graph_dot

    dot = graph_dot(_graph_doc(), top_n=1)  # keep only node A
    assert '"A"' in dot
    assert "->" not in dot  # both edges had an endpoint that was dropped


def test_graph_dot_escapes_quotes_in_titles() -> None:
    from magnetor.dashboard_data import graph_dot

    node = {"id": "A", "title": 'He said "hi"', "influence": 0.5}
    doc: dict[str, object] = {"nodes": [node], "edges": []}
    dot = graph_dot(doc)
    assert '"hi"' not in dot  # inner double-quotes were neutralised


def test_graph_dot_colours_traced_legs_distinctly() -> None:
    from magnetor.dashboard_data import graph_dot

    dot = graph_dot(_graph_doc(), traced={("B", "A"): "roots", ("C", "A"): "development"})
    assert '"B" -> "A" [color="#8e44ad", penwidth=3.0' in dot  # roots leg, purple
    assert '"C" -> "A" [color="#1f9d55", penwidth=3.0' in dot  # development leg, green


def test_graph_dot_outlines_highlighted_and_anchor_nodes() -> None:
    from magnetor.dashboard_data import graph_dot

    dot = graph_dot(_graph_doc(), highlighted=["B"], anchor="A")
    anchor_line = next(line for line in dot.splitlines() if line.strip().startswith('"A"'))
    highlighted_line = next(line for line in dot.splitlines() if line.strip().startswith('"B"'))
    assert "peripheries=2" in anchor_line  # anchor is doubled
    assert "peripheries=2" not in highlighted_line
    assert "penwidth=2.5" in highlighted_line


def test_graph_dot_pins_traced_nodes_below_the_top_n_cut() -> None:
    from magnetor.dashboard_data import graph_dot

    # C ranks last and would be cut by top_n=1, but it carries a traced edge:
    # dropping it would leave the pathway pointing at a node that was never drawn.
    dot = graph_dot(_graph_doc(), top_n=1, traced={("C", "A"): "development"})
    assert '"C"' in dot
    assert '"C" -> "A"' in dot
    assert '"B"' not in dot  # untraced and below the cut -> still pruned


def test_node_detail_reports_absent_values_as_unavailable() -> None:
    from magnetor.dashboard_data import node_detail

    lines = "\n".join(node_detail({"id": "A", "title": "T", "influence": 0.5}))
    assert "unavailable" in lines  # missing year/venue/CI never render as 0
    assert "0.500" in lines


def test_node_detail_never_exposes_an_abstract() -> None:
    from magnetor.dashboard_data import node_detail

    node = {"id": "A", "title": "T", "influence": 0.1, "abstract": "LEAKEDBODY"}
    assert "LEAKEDBODY" not in "\n".join(node_detail(node))


def test_graph_dot_draws_relation_layers_arrowless_and_dashed() -> None:
    from magnetor.dashboard_data import graph_dot

    doc = _graph_doc()
    doc["biblio_coupled"] = [["A", "C", 5]]
    dot = graph_dot(doc, layers=["biblio_coupled"])
    line = next(ln for ln in dot.splitlines() if "8d6e63" in ln)
    assert "style=dashed" in line
    assert "dir=none" in line  # a derived relation asserts no precedence
    # The citation backbone keeps its arrows and stays visually separate (I2).
    assert '"B" -> "A" [color="#8a8a8a", arrowsize=0.5];' in dot


def test_graph_dot_hides_the_backbone_but_keeps_traced_edges() -> None:
    from magnetor.dashboard_data import graph_dot

    dot = graph_dot(_graph_doc(), show_backbone=False, traced={("B", "A"): "link"})
    assert '"B" -> "A"' in dot  # the traced edge survives
    assert "#0b7285" in dot  # drawn as a connection
    assert '"C" -> "A"' not in dot  # every untraced citation is gone
    # Nodes are still emitted, so papers remain visible with no edges at all.
    assert '"C" [' in dot


def test_graph_dot_hiding_the_backbone_shrinks_the_layout() -> None:
    """Edge count is what makes the browser-side layout hang, so this is the lever."""
    from magnetor.dashboard_data import graph_dot

    full = graph_dot(_graph_doc()).count("->")
    bare = graph_dot(_graph_doc(), show_backbone=False).count("->")
    assert bare == 0
    assert full > bare


def test_graph_dot_caps_relation_edges() -> None:
    from magnetor.dashboard_data import graph_dot

    doc = _graph_doc()
    doc["co_cited"] = [["A", "B", 9], ["A", "C", 8], ["B", "C", 7]]
    dot = graph_dot(doc, layers=["co_cited"], max_relation_edges=2)
    assert dot.count("5c6bc0") == 2  # only the two strongest drawn
    assert '"A" -> "B"' in dot  # weight 9 kept
    assert '"B" -> "C"' not in dot  # weight 7 dropped


def test_graph_dot_omits_relation_layers_unless_requested() -> None:
    from magnetor.dashboard_data import graph_dot

    doc = _graph_doc()
    doc["biblio_coupled"] = [["A", "C", 5]]
    assert "8d6e63" not in graph_dot(doc)


def test_graph_dot_only_nodes_restricts_to_a_subgraph() -> None:
    from magnetor.dashboard_data import graph_dot

    dot = graph_dot(_graph_doc(), only_nodes=["A", "B"])
    assert '"C"' not in dot
    assert '"B" -> "A"' in dot
    assert '"C" -> "A"' not in dot  # pruned with its endpoint


def test_relation_rows_tolerate_graphs_without_the_layer() -> None:
    from magnetor.dashboard_data import relation_rows

    assert relation_rows(_graph_doc(), "biblio_coupled") == ()
    assert relation_rows({"co_cited": "nonsense"}, "co_cited") == ()
    assert relation_rows({"co_cited": [["A", "B", 4], ["bad"]]}, "co_cited") == (("A", "B", 4),)


def test_related_papers_ranks_by_weight_and_excludes_self() -> None:
    from magnetor.dashboard_data import related_papers

    doc: dict[str, object] = {"biblio_coupled": [["A", "B", 3], ["A", "C", 9], ["B", "C", 1]]}
    assert related_papers(doc, "A", "biblio_coupled") == (("C", 9), ("B", 3))
    assert related_papers(doc, "A", "biblio_coupled", limit=1) == (("C", 9),)


def test_node_detail_surfaces_integrity_and_review_flags() -> None:
    from magnetor.dashboard_data import node_detail

    lines = "\n".join(
        node_detail(
            {
                "id": "A", "title": "T", "influence": 0.9, "median_rank": 3,
                "lo_rank": 2, "hi_rank": 7, "stable": False,
                "is_retracted": True, "is_review": True,
            }
        )
    )
    assert "Retracted" in lines
    assert "Review article" in lines
    assert "2-7" in lines
    assert "unstable" in lines
