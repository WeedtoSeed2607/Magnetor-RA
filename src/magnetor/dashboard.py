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
from dataclasses import dataclass
from typing import Any

import streamlit as st

from magnetor.citations import SemanticScholarClient
from magnetor.config import all_domains, get_domain_config
from magnetor.dashboard_data import (
    GRAPH_DISCLAIMER,
    RELATION_LABELS,
    banner_lines,
    citation_url,
    frontier_feed,
    graph_dot,
    linked,
    load_trends,
    node_detail,
    paper_url,
    related_papers,
    relation_rows,
    search_access,
)
from magnetor.deepdive import DeepDiveResult, Path, build_deep_dive
from magnetor.embeddings.base import Embedder
from magnetor.embeddings.voyage import VoyageEmbedder
from magnetor.errors import MagnetorError
from magnetor.export import LineageSet, lineage_bundle, slugify, zip_bundle
from magnetor.graph import list_graphs, load_graph
from magnetor.harvest_jobs import (
    DONE,
    RUNNING,
    harvest_ui_enabled,
    job_for,
    log_tail,
    start_anchor,
    start_harvest,
)
from magnetor.indexing import open_index
from magnetor.narrowing import BROAD, assess
from magnetor.pathways import (
    DEFAULT_ALPHA,
    DEFAULT_ANCHOR_THRESHOLD,
    DEFAULT_FLOOR,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PATHS,
    INDIRECT,
    LINEAGE,
    ProgressionPath,
    connect,
    graph_view,
    highlight,
    path_edges,
    progression_paths,
)
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
    _render_harvest_launcher()
    graphs = list_graphs()
    if not graphs:
        hint = (
            "No graphs yet — harvest one above, or run "
            '`magnetor harvest "<your question>"`.'
            if harvest_ui_enabled()
            else 'No graphs yet — build one with `magnetor harvest "<your question>"`.'
        )
        st.info(hint)
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
    _render_breadth(document)

    raw_nodes = document.get("nodes")
    listed = raw_nodes if isinstance(raw_nodes, list) else []
    all_nodes: list[dict[str, Any]] = [n for n in listed if isinstance(n, dict)]
    node_by_id = {str(n.get("id")): n for n in all_nodes}

    traced, highlighted, anchor = _render_pathway_controls(document, choice)
    inspect = _render_connection_controls(document, all_nodes, choice)
    if inspect.hide_paths:
        traced, highlighted, anchor = {}, (), None
    traced = {**traced, **inspect.edges}
    highlighted = tuple(dict.fromkeys((*highlighted, *inspect.nodes)))

    top_n = st.slider("Nodes to draw", min_value=10, max_value=80, value=40, step=5)
    dot = graph_dot(
        document,
        top_n=top_n,
        traced=traced,
        highlighted=highlighted,
        anchor=anchor,
        layers=inspect.layers,
        only_nodes=inspect.only_nodes,
        show_backbone=inspect.show_backbone,
    )
    st.graphviz_chart(dot, width="stretch")
    drawn = dot.count(" -> ")
    if drawn > 250:
        # The browser lays this out; past a few hundred edges it stops being
        # interactive, and the cause is never obvious from the picture itself.
        st.caption(
            f"⚠️ {drawn} edges drawn — layout runs in your browser and gets slow "
            "here. Switch **Edges to draw** to *Only what I'm tracing*, or lower "
            "**Nodes to draw**."
        )
    legend = []
    if any(leg in ("roots", "development") for leg in traced.values()):
        legend.append(
            "🟣 **purple** back toward roots · 🟢 **green** forward into later work"
        )
    if any(leg == "link" for leg in traced.values()):
        legend.append("🔵 **teal** the connection between the papers you picked")
    if "biblio_coupled" in inspect.layers:
        legend.append("**brown dashed** shares references (no citation between them)")
    if "co_cited" in inspect.layers:
        legend.append("**indigo dotted** cited together by the same later papers")
    if legend:
        st.caption("Edge key — " + " · ".join(legend) + ".")
    st.info(GRAPH_DISCLAIMER)

    _render_node_detail(document, all_nodes)

    st.markdown("**Most influential — traceable pathway**")
    for node in all_nodes[:10]:
        st.markdown(_influence_line(node), unsafe_allow_html=True)
    if anchor and anchor in node_by_id:
        st.caption(
            "Origin is the earliest *relevant* paper **in available data** — with "
            "boundary leakage the field's true origin is very likely outside this set."
        )


def _render_harvest_launcher() -> None:
    """Start a harvest from the page instead of a terminal (ADR-0006 I5 preserved).

    The button only *spawns* the CLI; the request returns immediately and status
    comes from polling files. Hidden entirely unless ``MAGNETOR_ENABLE_HARVEST_UI``
    is set, so a hosted deployment can never spawn a job or spend quota.
    """
    if not harvest_ui_enabled():
        return
    with st.expander("Build a new graph (runs in the background)"):
        st.caption(
            "Builds a new Evidence Graph from OpenAlex. This takes minutes, so it "
            "runs as a background job — you can keep using the dashboard, and the "
            "graph appears in the picker below when it finishes."
        )
        mode = st.radio(
            "Start from",
            _BUILD_MODES,
            horizontal=True,
            key="harvest_mode",
            help="A question gathers papers matching a keyword search. A paper "
            "gathers its citation neighbourhood instead — what it was built on and "
            "what has built on it since.",
        )
        anchored = mode == _BUILD_MODES[1]
        query = st.text_input(
            "Paper (DOI, OpenAlex id, or a link)" if anchored else "Research question",
            key="harvest_new_query",
            placeholder="10.1037/0033-295x.108.4.814"
            if anchored
            else "e.g. topological quantum error correction",
        )
        limit = st.slider("Papers to fetch", 50, 500, 200, 50, key="harvest_limit")
        if anchored:
            backward = st.slider(
                "Hops back toward antecedents", 1, 3, 2, key="anchor_backward",
                help="Cheap — references arrive with each paper already fetched.",
            )
            fanout = st.slider(
                "Also expand this many citing papers forward", 0, 10, 0,
                key="anchor_forward",
                help="0 expands the seed alone. Each extra paper costs another query, "
                "so raise this only when the neighbourhood is too thin.",
            )
            st.caption(
                "Note: the seed will rank top by construction — the set is built from "
                "papers citing it. Read the ranking among the *other* papers."
            )
        else:
            expand = st.slider(
                "Snowball rounds", 0, 3, 2, key="harvest_expand",
                help="Extra passes pulling in foundational papers the seed set cites. "
                "Cuts boundary leakage but lengthens the run.",
            )
        if st.button("Start build", disabled=not query.strip(), key="harvest_start"):
            if anchored:
                start_anchor(
                    query.strip(), limit=limit, backward=backward, forward_fanout=fanout
                )
            else:
                start_harvest(query.strip(), limit=limit, expand=expand)
            st.session_state["harvest_watching"] = query.strip()
            st.rerun()
        _render_harvest_status()


def _render_harvest_status() -> None:
    """Poll the watched job's exit marker and show its log tail."""
    watching = str(st.session_state.get("harvest_watching") or "")
    if not watching:
        return
    job = job_for(watching)
    if job is None:
        return
    if job.status == RUNNING:
        st.info(f"Harvesting {watching!r} — started {job.started_at[:19]}Z")
        st.button("Refresh status", key="harvest_refresh")
    elif job.status == DONE:
        st.success(f"Finished: {watching!r}. Select it in the picker below.")
    else:
        st.error(
            f"Harvest failed (exit {job.exit_code}). The log below has the reason — "
            "a query with no OpenAlex results is the usual cause."
        )
    tail = log_tail(job)
    if tail:
        st.code("\n".join(tail))


def _render_breadth(document: dict[str, Any]) -> None:
    """Say whether the query was too broad, and what would narrow it (concern 2).

    Suggestions are only shown when the graph actually reads as broad. On a
    focused graph the same machinery still returns phrases, but recommending a
    narrowing nobody needs is how a useful signal becomes noise.
    """
    report = assess(document)
    stats = (
        f"{report.pair_coverage:.0%} of paper pairs are citation-linked · "
        f"{report.components} component(s) · "
        f"{report.largest_component_share:.0%} in the largest"
    )
    if not report.is_broad:
        st.caption(f"Query looks **focused** — {stats}.")
        return

    label = "broad" if report.verdict == BROAD else "scattered"
    st.warning(
        f"This query looks **{label}** — {stats}. The papers it gathered largely "
        "do not cite one another, which usually means several literatures were "
        "collected side by side rather than one lineage."
    )
    if not report.suggestions:
        st.caption("No sub-topic in this graph was cohesive enough to suggest.")
        return
    st.markdown("**Narrower questions this graph already contains:**")
    for suggestion in report.suggestions:
        cols = st.columns([5, 1])
        cols[0].markdown(
            f"- **{suggestion.phrase}** — {suggestion.papers} papers "
            f"({suggestion.share:.0%}), citing each other "
            f"{suggestion.cohesion:.1f}x more densely than the graph average"
        )
        if harvest_ui_enabled() and cols[1].button(
            "Harvest", key=f"narrow_{suggestion.phrase}", help=f"Harvest {suggestion.phrase!r}"
        ):
            start_harvest(suggestion.phrase)
            st.session_state["harvest_watching"] = suggestion.phrase
            st.rerun()
    st.caption(
        "Cohesion is measured against this graph only, and the broad/focused "
        "cutoffs are declared conventions rather than calibrated thresholds."
    )


def _influence_line(node: dict[str, Any]) -> str:
    """One bullet in the influence list: title link, flags, and score components."""
    flags = " 🔴" if node.get("is_retracted") else ""
    flags += " 🟠" if node.get("stable") is False else ""
    ci = ""
    if node.get("lo_rank"):
        ci = f" · rank {node.get('lo_rank')}-{node.get('hi_rank')}"
    title = linked(str(node.get("title") or node.get("id")), str(node.get("url") or "") or None)
    return (
        f"- {title}{flags}  \n"
        f"  <small>influence {float(node.get('influence') or 0):.2f} · "
        f"in-set citations {node.get('in_degree', 0)}{ci}</small>"
    )


def _advance_alternative(key: str, total: int) -> None:
    """Step the k-best cursor. A callback, so the new index is live on the rerun."""
    st.session_state[key] = (int(st.session_state.get(key, 0)) + 1) % max(1, total)


def _render_pathway_controls(
    document: dict[str, object], choice: int
) -> tuple[dict[tuple[str, str], str], tuple[str, ...], str | None]:
    """L5.2 + L5.3: graph-scoped search, alpha blend, and the k-best "Next" cursor.

    Returns what the renderer needs — traced edges, highlighted nodes, anchor —
    all empty until a sub-query is entered, so the default view is unchanged.
    """
    st.markdown("**Trace a pathway inside this graph**")
    subquery = st.text_input(
        "Keyword or sub-question",
        key=f"graph_subquery_{choice}",
        placeholder="e.g. surface code decoders",
        help=(
            "Searches within the graph already loaded above — it does not harvest "
            "anything new. Matching is on titles only (abstracts are deliberately "
            "not stored in a graph artifact), so try the field's own vocabulary."
        ),
    )
    if not subquery.strip():
        return {}, (), None

    alt_key = f"graph_path_alt_{choice}"
    last_key = f"graph_lastq_{choice}"
    if st.session_state.get(last_key) != subquery:
        st.session_state[last_key] = subquery
        st.session_state[alt_key] = 0

    alpha = st.slider(
        "Weighting: influence <-> keyword (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_ALPHA,
        step=0.05,
        help=(
            "alpha=1 follows the keyword alone; alpha=0 follows influence alone. "
            "Lower it to lengthen the roots leg, since foundational papers rarely "
            "match a keyword."
        ),
    )
    with st.expander("Advanced path controls"):
        threshold = st.slider(
            "Origin relevance threshold", 0.0, 1.0, DEFAULT_ANCHOR_THRESHOLD, 0.01,
            help="How relevant a paper must be to qualify as the origin. Lower it and a "
            "barely-relevant older paper can win the origin.",
        )
        floor = st.slider(
            "Step weight floor", 0.0, 0.5, DEFAULT_FLOOR, 0.01,
            help="The walk continues until no next paper clears this weight (open-ended).",
        )
        depth = st.slider("Max steps per leg", 1, 12, DEFAULT_MAX_DEPTH)

    view = graph_view(document)
    found = highlight(view, subquery)
    if not any(score > 0 for score in found.relevance.values()):
        st.warning("No paper title in this graph matches those terms — nothing to trace.")
        return {}, (), None

    paths = progression_paths(
        view, subquery, alpha=alpha, threshold=threshold, floor=floor, max_depth=depth
    )
    if not paths:
        st.info(
            "Relevant papers are highlighted, but none clears the origin threshold — "
            "lower it in Advanced to trace a path."
        )
        return {}, found.selected, None

    index = int(st.session_state.get(alt_key, 0)) % len(paths)
    path = paths[index]
    st.caption(f"Pathway {index + 1} of {len(paths)} · {len(path.nodes)} papers")
    st.button(
        "Next alternative →",
        on_click=_advance_alternative,
        args=(alt_key, len(paths)),
        help="Step to the next-most-probable progression.",
    )
    _render_path_chain(path, document)
    _offer_download(
        document,
        [LineageSet(label=f"Progression for {subquery!r}", kind="progression",
                    routes=(path.nodes,))],
        key=f"dl_path_{choice}",
        label="Download this progression",
    )
    return path_edges(path), found.selected, path.anchor


def _render_path_chain(path: ProgressionPath, document: dict[str, object]) -> None:
    """List the traced chain in reading order: oldest root → origin → latest work."""
    raw_nodes = document.get("nodes")
    entries = [n for n in (raw_nodes if isinstance(raw_nodes, list) else []) if isinstance(n, dict)]
    by_id = {str(n.get("id")): n for n in entries}
    for node_id in path.nodes:
        node = by_id.get(node_id)
        if node is None:
            continue
        title = linked(str(node.get("title") or node_id), str(node.get("url") or "") or None)
        is_origin = node_id == path.anchor
        marker = " <- **origin (earliest relevant in available data)**" if is_origin else ""
        year = node.get("year") or "?"
        st.markdown(f"- {year} · {title}{marker}")


#: Edge-display modes. Graphviz lays out in the browser and its cost tracks edge
#: count, so hiding the backbone is the main lever on responsiveness as well as
#: on legibility.
_EDGE_MODES = ("All citations", "Only what I'm tracing", "No edges at all")

#: The operator's two entry points (concern 5): start from a question when you do
#: not know where to begin, or from a paper you already have to trace its evolution.
_BUILD_MODES = ("A research question", "A specific paper")


@dataclass(frozen=True, slots=True)
class _InspectState:
    """What the inspect panel contributes to the drawing."""

    hide_paths: bool
    show_backbone: bool
    edges: dict[tuple[str, str], str]
    nodes: tuple[str, ...]
    only_nodes: tuple[str, ...] | None
    layers: tuple[str, ...]


def _offer_download(
    document: dict[str, Any], sets: list[LineageSet], *, key: str, label: str
) -> None:
    """Package traced routes as a citation folder the researcher can keep.

    Metadata and resolvable links, never the papers themselves — see
    ``export.py`` for why. Built eagerly because Streamlit's download button
    needs the bytes up front; the bundle is small enough that this is free.
    """
    if not any(item.routes for item in sets):
        return
    files = lineage_bundle(document, sets)
    st.download_button(
        label,
        data=zip_bundle(files),
        file_name=f"{slugify(str(document.get('query', '')))}-lineages.zip",
        mime="application/zip",
        key=key,
        help="A folder of citations: reading order, CSV, BibTeX, and the provenance. "
        "Not the PDFs — most are paywalled, so each entry carries its DOI link.",
    )


def _short(node: dict[str, Any] | None, node_id: str, width: int = 46) -> str:
    if node is None:
        return node_id
    title = str(node.get("title") or node_id)
    return title if len(title) <= width else title[: width - 1] + "…"


def _render_connection_controls(
    document: dict[str, Any], all_nodes: list[dict[str, Any]], choice: int
) -> _InspectState:
    """Pick papers, ask whether they are connected, and overlay the answer.

    Selection rather than clicking the image: ``st.graphviz_chart`` emits a static
    picture with no click events. The gesture differs from the ask; the question
    answered is the same one.
    """
    by_id = {str(n.get("id")): n for n in all_nodes}
    st.markdown("**Inspect specific papers**")
    mode = st.radio(
        "Edges to draw",
        _EDGE_MODES,
        horizontal=True,
        key=f"edge_mode_{choice}",
        help="Hiding the citation backbone clears the clutter and is also what makes "
        "the graph responsive — the browser lays it out, and ~213 citations plus the "
        "relation layers is past the point where that stays interactive.",
    )
    show_backbone = mode == _EDGE_MODES[0]
    hide_paths = mode == _EDGE_MODES[2]

    labels = {
        str(n.get("id")): f"{i + 1}. {_short(n, str(n.get('id')), 70)}"
        for i, n in enumerate(all_nodes)
    }
    picked = st.multiselect(
        "Papers to connect (pick two or more)",
        options=list(labels),
        format_func=lambda nid: labels[nid],
        key=f"connect_pick_{choice}",
        help="The check runs as soon as two are selected.",
    )

    available = [k for k in ("biblio_coupled", "co_cited") if relation_rows(document, k)]
    layers: list[str] = []
    if available:
        layers = st.multiselect(
            "Show derived relation layers",
            options=available,
            format_func=lambda k: RELATION_LABELS[k],
            key=f"layers_{choice}",
            help="Relations computed from reference lists, not from citations between "
            "these papers. Drawn dashed and arrowless — they never join the citation "
            "backbone, and they do not affect any influence score.",
        )
    else:
        st.caption(
            "Derived relation layers are not stored in this graph — re-harvest the "
            "question to compute co-citation and bibliographic coupling."
        )

    if len(picked) < 2:
        return _InspectState(hide_paths, show_backbone, {}, (), None, tuple(layers))

    expand = st.checkbox(
        "Show every lineage, not just the shortest",
        value=True,
        key=f"connect_expand_{choice}",
        help="Citation lineages branch and rejoin, so the shortest chain alone "
        "understates the link. Capped at 8 routes per pair.",
    )
    report = connect(graph_view(document), picked, expand=expand)
    for link in report.pairs:
        left, right = _short(by_id.get(link.a), link.a), _short(by_id.get(link.b), link.b)
        steps = max(0, len(link.path) - 1)
        if link.kind == LINEAGE:
            st.success(f"**{left}** → **{right}** — citation lineage, {steps} step(s).")
        elif link.kind == INDIRECT:
            st.info(
                f"**{left}** ~ **{right}** — no lineage either way; joined through "
                f"{steps - 1} shared relative(s)."
            )
        else:
            st.warning(
                f"**{left}** / **{right}** — not connected *within this harvested "
                "slice*. That is a statement about the harvest, not about the field."
            )
        routes = link.lineages or ((link.path,) if link.path else ())
        for index, route in enumerate(routes, start=1):
            chain = "  →  ".join(
                f"{by_id.get(n, {}).get('year') or '?'} {_short(by_id.get(n), n, 34)}"
                for n in route
            )
            prefix = f"**{index}.** " if len(routes) > 1 else ""
            st.caption(f"{prefix}{chain}")
        if len(routes) >= DEFAULT_MAX_PATHS:
            st.caption(
                f"_Showing the {DEFAULT_MAX_PATHS} shortest lineages; there may be more._"
            )

    _offer_download(
        document,
        [
            LineageSet(
                label=f"{_short(by_id.get(link.a), link.a, 40)} -> "
                f"{_short(by_id.get(link.b), link.b, 40)}",
                kind=link.kind,
                routes=link.lineages or ((link.path,) if link.path else ()),
            )
            for link in report.pairs
        ],
        key=f"dl_connect_{choice}",
        label="Download these lineages",
    )

    only_nodes: tuple[str, ...] | None = None
    if st.checkbox(
        "Show only these papers and the chain between them",
        key=f"connect_only_{choice}",
        help="Drops every unrelated node so the connection is readable.",
    ):
        only_nodes = tuple(dict.fromkeys((*picked, *report.nodes)))
    return _InspectState(
        hide_paths=hide_paths,
        show_backbone=show_backbone,
        edges=report.edges,
        nodes=tuple(dict.fromkeys((*picked, *report.nodes))),
        only_nodes=only_nodes,
        layers=tuple(layers),
    )


def _render_node_detail(document: dict[str, Any], all_nodes: list[dict[str, Any]]) -> None:
    """L5.1(1): the node detail panel — identity, link out, per-metric breakdown.

    Selection rather than click-to-inspect: ``st.graphviz_chart`` renders a static
    image with no click events, and swapping renderers was declined in favour of
    keeping the panel dependency-free.
    """
    if not all_nodes:
        return
    with st.expander("Inspect a paper (full title · link · influence breakdown)"):
        labels = [
            f"{i + 1}. {str(n.get('title') or n.get('id'))[:80]}"
            for i, n in enumerate(all_nodes)
        ]
        picked = st.selectbox(
            "Paper", range(len(all_nodes)), format_func=lambda i: labels[i], key="graph_node_detail"
        )
        node = all_nodes[picked]
        url = str(node.get("url") or "") or None
        st.markdown(f"### {linked(str(node.get('title') or node.get('id')), url)}")
        for line in node_detail(node):
            st.markdown(line)
        if url:
            st.markdown(f"[Open the paper ↗]({url})")
        _render_related(document, all_nodes, str(node.get("id")))


def _render_related(
    document: dict[str, Any], all_nodes: list[dict[str, Any]], node_id: str
) -> None:
    """Derived neighbours of one paper — the cross-niche view.

    These are papers standing on the same foundations, which is exactly the set a
    keyword search cannot reach: most of them never cite this paper and it never
    cites them.
    """
    by_id = {str(n.get("id")): n for n in all_nodes}
    cites = {
        (str(e[0]), str(e[1]))
        for e in (document.get("edges") or [])
        if isinstance(e, list) and len(e) == 2
    }
    for kind in ("biblio_coupled", "co_cited"):
        related = related_papers(document, node_id, kind)
        if not related:
            continue
        st.markdown(f"**{RELATION_LABELS[kind].capitalize()}:**")
        for other_id, weight in related:
            other = by_id.get(other_id)
            title = linked(
                str((other or {}).get("title") or other_id),
                str((other or {}).get("url") or "") or None,
            )
            direct = (node_id, other_id) in cites or (other_id, node_id) in cites
            note = "" if direct else " · _no citation between them_"
            st.markdown(f"- {title} <small>(weight {weight}{note})</small>", unsafe_allow_html=True)


# Streamlit runs the entry script in a module named "__main__" and re-executes
# it top-to-bottom on every interaction, so calling main() here re-renders each
# rerun. Guarded so importing this module (e.g. from streamlit_app.py) does NOT
# double-run — that entry calls main() itself.
if __name__ == "__main__":
    main()
