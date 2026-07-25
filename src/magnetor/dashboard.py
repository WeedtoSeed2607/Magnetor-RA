"""Magnetor dashboard (Spec Section 11) — Streamlit.

Run via ``magnetor dashboard`` (or ``python -m streamlit run`` on this file).

Four panels per Spec 11:
- Topic-Trend Banner (Branch A) — statistic-anchored, replaces "mood of season"
- Primary Viewport (Branch B) — Anchor Paper Card or Field Map; domain labelled
- Frontier Feed (Branch A) — anomaly alerts, hot clusters, recent papers
- Sentiment panel (optional) — the module is deferred, so it is not shown

Rendering only; the testable view-model logic lives in ``dashboard_data.py``.
"""

from __future__ import annotations

import os

import streamlit as st

from magnetor.citations import SemanticScholarClient
from magnetor.config import all_domains, get_domain_config
from magnetor.dashboard_data import (
    banner_lines,
    citation_url,
    frontier_feed,
    graph_dot,
    linked,
    load_trends,
    paper_url,
    search_access,
)
from magnetor.deepdive import DeepDiveResult, Path, build_deep_dive
from magnetor.embeddings.base import Embedder
from magnetor.embeddings.voyage import VoyageEmbedder
from magnetor.errors import MagnetorError
from magnetor.graph import list_graphs, load_graph
from magnetor.indexing import open_index
from magnetor.router import CrossDomainRouter
from magnetor.types import Domain


@st.cache_resource
def _embedder() -> Embedder:
    return VoyageEmbedder()


@st.cache_resource
def _router() -> CrossDomainRouter:
    embedder = _embedder()
    indices = {d: open_index(get_domain_config(d), embedder) for d in all_domains()}
    return CrossDomainRouter(indices)


def _deep_dive(query: str, domain: Domain | None) -> DeepDiveResult:
    return build_deep_dive(
        _embedder(), _router(), query, SemanticScholarClient(), domain=domain
    )


def _render_banner(active: Domain) -> None:
    st.subheader("Topic-Trend Banner")
    trends = load_trends(active)
    lines = banner_lines(trends)
    if lines:
        for line in lines:
            st.markdown(f"- {line}")
    else:
        st.info(f"No trends yet for **{active.value}** — run `magnetor trends {active.value}`.")


def _search_unlocked() -> bool:
    """Password-gate the live deep-dive (Spec 7.2) when hosted publicly.

    A password set via ``MAGNETOR_SEARCH_PASSWORD`` (Streamlit secret in the
    cloud) protects the query box so a public link can't drain the API key.
    No password configured → open, which is the local-dev default.
    """
    expected = os.environ.get("MAGNETOR_SEARCH_PASSWORD")
    already_ok = bool(st.session_state.get("_search_ok"))
    entered = None
    if expected and not already_ok:
        entered = st.sidebar.text_input("🔒 Search password", type="password")
    unlocked, error = search_access(expected, entered, already_ok)
    if unlocked and expected:
        st.session_state["_search_ok"] = True
    if error:
        st.sidebar.error(error)
    return unlocked


def _render_primary(active: Domain, query: str, force_domain: bool, unlocked: bool) -> None:
    st.subheader("Primary Viewport")
    if not query:
        st.caption("Enter a query in the sidebar to run a Branch B deep-dive.")
        return
    if not unlocked:
        st.info("🔒 Search is password-protected — enter the password in the sidebar.")
        return
    try:
        result = _deep_dive(query, active if force_domain else None)
    except MagnetorError as exc:
        st.error(str(exc))
        return
    if result.path is None:
        st.warning("No results — has this domain been embedded? (`magnetor embed`)")
        return

    label = result.domain.value if result.domain else "-"
    score = result.top_score if result.top_score is not None else 0.0
    threshold = result.threshold if result.threshold is not None else 0.0
    st.caption(
        f"domain **{label}**  ·  path **{result.path.value}**  ·  "
        f"score {score:.3f} vs threshold {threshold:.3f}"
    )
    if result.path is Path.ANCHOR_LOCK and result.anchor is not None:
        _render_anchor(result)
    elif result.path is Path.FIELD_MAP and result.field_map is not None:
        _render_field_map(result)


def _render_anchor(result: DeepDiveResult) -> None:
    anchor = result.anchor
    assert anchor is not None
    st.markdown("### 🔒 Anchor")
    st.markdown(f"**{linked(anchor.paper.title, paper_url(anchor.paper))}**")
    st.write(anchor.paper.abstract or "(no abstract)")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Cited by ({len(anchor.forward)})**")
        for c in anchor.forward[:10]:
            year = f" ({c.year})" if c.year else ""
            st.markdown(f"- {linked(c.title, citation_url(c))}{year}")
    with right:
        st.markdown(f"**References ({len(anchor.backward)})**")
        for c in anchor.backward[:10]:
            year = f" ({c.year})" if c.year else ""
            st.markdown(f"- {linked(c.title, citation_url(c))}{year}")


def _render_field_map(result: DeepDiveResult) -> None:
    field_map = result.field_map
    assert field_map is not None
    st.markdown("### 🗺️ Field Map — competing positions")
    for pos in field_map.positions:
        with st.expander(f"[{pos.rank}] {pos.score:.3f} — {pos.paper.title}"):
            url = paper_url(pos.paper)
            if url:
                st.markdown(f"🔗 {linked('Open paper', url)}")
            st.write(pos.paper.abstract or "(no abstract)")


def _render_frontier(active: Domain) -> None:
    st.subheader("Frontier Feed")
    feed = frontier_feed(active, load_trends(active))
    st.markdown("**Emerging terms** (volume-normalized)")
    if feed.anomalies:
        for a in feed.anomalies[:8]:
            raw = a.get("delta", 0.0)
            delta = float(raw) if isinstance(raw, (int, float)) else 0.0
            st.markdown(f"- `{a.get('term')}`  (+{delta:.2f}/doc)")
    else:
        st.caption("none")
    st.markdown("**Hot clusters**")
    if feed.hot_clusters:
        for topic in feed.hot_clusters[:5]:
            raw_kws = topic.get("keywords")
            keywords = raw_kws if isinstance(raw_kws, list) else []
            kws = ", ".join(str(k) for k in keywords[:5])
            st.markdown(f"- {kws}")
    else:
        st.caption("none")
    st.markdown("**Recent papers**")
    for paper in feed.recent:
        st.markdown(f"- {linked(paper.title, paper_url(paper))}")


def main() -> None:
    st.set_page_config(page_title="Magnetor", layout="wide")
    st.title("Magnetor — Domain-Aware Research Platform")
    st.caption("Silo the data, federate the answer.")

    domain_token = st.sidebar.selectbox("Active domain", [d.value for d in all_domains()])
    active = Domain(domain_token)
    query = st.sidebar.text_input("Deep-dive query (Branch B)")
    force_domain = st.sidebar.checkbox("Force this domain (skip routing)", value=False)
    unlocked = _search_unlocked()

    _render_banner(active)
    st.divider()
    primary, secondary = st.columns([2, 1])
    with primary:
        _render_primary(active, query, force_domain, unlocked)
    with secondary:
        _render_frontier(active)
    # Sentiment panel (Spec 11) omitted: the sentiment module is optional and
    # not enabled, and must never be blended into the Topic-Trend Banner.
    st.divider()
    _render_evidence_graph()


def _render_evidence_graph() -> None:
    """Branch C (ADR-0006): the query-relative influence node map."""
    st.subheader("Evidence Graph — query-relative influence (Branch C)")
    graphs = list_graphs()
    if not graphs:
        st.info('No graphs yet — build one with `magnetor harvest "<your question>"`.')
        return

    labels = [f"{g['query']}  ·  {g['n_nodes']} papers" for g in graphs]
    choice = st.selectbox("Harvested question", range(len(graphs)), format_func=lambda i: labels[i])
    document = load_graph(str(graphs[choice]["query"]))
    if not document:
        st.warning("Could not load that graph.")
        return

    leak = float(document.get("boundary_leakage") or 0.0)
    n_nodes = len(document.get("nodes", []))
    when = str(document.get("generated_at", "?"))[:10]
    st.caption(
        f"{n_nodes} nodes · generated {when} · node size = influence · "
        "🔴 retracted · 🟠 rank unstable"
    )
    if leak >= 0.5:
        st.warning(
            f"Boundary leakage {leak:.0%}: most citations point outside the harvested set, "
            "so this is a partial slice of the lineage (snowball expansion pending)."
        )

    top_n = st.slider("Nodes to draw", min_value=10, max_value=80, value=40, step=5)
    st.graphviz_chart(graph_dot(document, top_n=top_n), width="stretch")

    st.markdown("**Most influential — traceable pathway**")
    for node in document.get("nodes", [])[:10]:
        flags = " 🔴" if node.get("is_retracted") else ""
        flags += " 🟠" if node.get("stable") is False else ""
        ci = ""
        if node.get("lo_rank"):
            ci = f" · rank {node.get('lo_rank')}-{node.get('hi_rank')}"
        title = linked(str(node.get("title") or node.get("id")), str(node.get("url") or "") or None)
        st.markdown(
            f"- {title}{flags}  \n"
            f"  <small>influence {float(node.get('influence') or 0):.2f} · "
            f"in-set citations {node.get('in_degree', 0)}{ci}</small>",
            unsafe_allow_html=True,
        )


# Streamlit runs the entry script in a module named "__main__" and re-executes
# it top-to-bottom on every interaction, so calling main() here re-renders each
# rerun. Guarded so importing this module (e.g. from streamlit_app.py) does NOT
# double-run — that entry calls main() itself.
if __name__ == "__main__":
    main()
