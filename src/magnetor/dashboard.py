"""Magnetor dashboard (Spec Section 11) — Streamlit.

Run via ``magnetor dashboard`` (or ``python -m streamlit run`` on this file).

Three pages:
- Trends (Branch A) — Topic-Trend Banner and Frontier Feed, statistic-anchored
- Evidence Graph (Branch C) — harvest, trace, inspect, export
- Verdict layer — Lakatosian appraisal of one paper at a time

Branch B's Primary Viewport was removed from the dashboard as redundant; the
deep-dive itself is untouched and still runs from ``magnetor deepdive``.
The Sentiment panel (Spec 11, optional) remains deferred and is not shown.

Rendering only; the testable view-model logic lives in ``dashboard_data.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from magnetor.coding import (
    ALPHA_FIRM,
    ALPHA_FLOOR,
    CONTEXT,
    DEFAULT_DEGENERATION_WINDOW,
    DEFEATER,
    FIELD_GUIDE,
    FIELDS,
    FIRM,
    GATE,
    STANDING_PLAIN,
    UNRELIABLE,
    UNSET,
    UNVALIDATED,
    USABLE,
    VERDICT_PLAIN,
    QualifiedVerdict,
    current_coder,
    fields_by_role,
    judge,
    prefill,
    programme_verdict,
    qualify,
    reliability_report,
    to_csv,
)
from magnetor.coding import load as load_coding
from magnetor.coding import record as record_coding
from magnetor.config import (
    all_domains,
    get_domain_config,
    load_env_file,
)
from magnetor.dashboard_data import (
    GRAPH_DISCLAIMER,
    RELATION_LABELS,
    banner_lines,
    frontier_feed,
    graph_dot,
    linked,
    load_trends,
    node_detail,
    paper_url,
    papers_for_term,
    papers_for_topic,
    related_papers,
    relation_rows,
)
from magnetor.diagnostics import Diagnosis, diagnose_no_trends
from magnetor.errors import MagnetorError
from magnetor.export import LineageSet, lineage_bundle, slugify, zip_bundle
from magnetor.facets import (
    UNCLASSIFIED,
    cross_facet_neighbours,
    facet_counts,
    node_facets,
    unclassified_breakdown,
)
from magnetor.gaps import facet_gaps, read_gaps
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
from magnetor.resources import DomainStore
from magnetor.types import Domain, Paper

# Pick up the project's .env before anything asks for an API key. Only mag.cmd
# used to do this, so a dashboard started any other way had the corpus but no key
# — and reported it as "no results", which points at the wrong problem entirely.
load_env_file()


def _record_count(domain: Domain) -> int | None:
    """Papers held for a subject, or ``None`` if the store could not be read."""
    try:
        config = get_domain_config(domain)
        return DomainStore(domain, config.storage_dir).record_count()
    except (OSError, MagnetorError):
        return None


def _render_banner(active: Domain) -> None:
    st.subheader("Topic-Trend Banner")
    trends = load_trends(active)
    lines = banner_lines(trends)
    if lines:
        for line in lines:
            st.markdown(f"- {line}")
        return
    # An empty banner used to always say "run magnetor trends", which does
    # nothing when the subject holds no papers. Diagnose it instead.
    _render_diagnosis(
        diagnose_no_trends(
            records=_record_count(active), trends_present=trends is not None
        )
    )


def _render_diagnosis(found: Diagnosis) -> None:
    """Show the conclusion and, underneath it, what was actually looked at.

    The evidence is on screen rather than in a log because that is what makes the
    claim checkable. A reader who disagrees with the conclusion can see which
    observation produced it.
    """
    if found.is_identified:
        st.warning(f"**{found.headline}**\n\n{found.remedy}")
    else:
        st.info(f"**{found.headline}**\n\n{found.remedy}")
    with st.expander("What was checked"):
        for check in found.evidence:
            mark = "✅" if check.observed else "❌" if check.conclusive else "❔"
            st.markdown(f"- {mark} **{check.name}** — {check.detail}")
        if not found.is_identified:
            st.caption(
                "No cause is being named because the checks above are incomplete. "
                "Guessing here is what produced misleading messages before."
            )


def _render_frontier(active: Domain) -> None:
    st.subheader("Frontier Feed")
    feed = frontier_feed(active, load_trends(active))
    st.markdown("**Words gaining ground**")
    if feed.anomalies:
        for a in feed.anomalies[:8]:
            raw = a.get("delta", 0.0)
            delta = float(raw) if isinstance(raw, (int, float)) else 0.0
            term = str(a.get("term") or "")
            with st.expander(f"{term}  ·  +{delta:.2f} per paper"):
                _render_paper_list(papers_for_term(active, term))
    else:
        st.caption("none")
    st.markdown("**Themes gaining ground**")
    if feed.hot_clusters:
        for topic in feed.hot_clusters[:5]:
            raw_kws = topic.get("keywords")
            keywords = [str(k) for k in (raw_kws if isinstance(raw_kws, list) else [])]
            kws = ", ".join(keywords[:5])
            with st.expander(kws or "(unnamed theme)"):
                matched = papers_for_topic(active, keywords[:6])
                if matched:
                    st.caption(
                        "Papers using several of these words — more matched words "
                        "first. Themes have no stored paper list, so this is a "
                        "keyword match, not the model's own grouping."
                    )
                    for paper, hits in matched:
                        st.markdown(
                            f"- {linked(paper.title, paper_url(paper))} "
                            f"<small>({hits} of these words)</small>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No stored paper uses more than one of these words.")
    else:
        st.caption("none")
    st.markdown("**Most recently collected**")
    st.caption("Newest arrivals, regardless of whether they are gaining attention.")
    for paper in feed.recent:
        st.markdown(f"- {linked(paper.title, paper_url(paper))}")


def _render_paper_list(papers: tuple[Paper, ...]) -> None:
    if not papers:
        st.caption(
            "No stored paper uses this word — it may come from an abstract the "
            "trend engine read but this search does not reach."
        )
        return
    for paper in papers:
        st.markdown(f"- {linked(paper.title, paper_url(paper))}")


def main() -> None:
    """Three workspaces rather than one page.

    They answer different questions at different moments — trends survey a
    subject, the graph traces one lineage, the verdict layer appraises a single
    paper. Stacking them made every panel compete with panels the reader was not
    using; splitting them lets each carry its own controls.
    """
    st.set_page_config(page_title="Magnetor", layout="wide")
    st.navigation(
        [
            st.Page(_page_trends, title="Trends", icon="📈", default=True),
            st.Page(_page_graph, title="Evidence Graph", icon="🕸️"),
            st.Page(_page_verdict, title="Verdict layer", icon="⚖️"),
        ]
    ).run()


def _active_domain() -> Domain:
    """Domain picker, rendered by whichever page needs one."""
    token = st.sidebar.selectbox(
        "Active domain", [d.value for d in all_domains()], key="active_domain"
    )
    return Domain(token)


def _lead(text: str) -> None:
    """The one sentence a reader must not be able to miss."""
    st.markdown(_LEAD_CSS, unsafe_allow_html=True)
    st.markdown(f'<p class="mag-lead">{text}</p>', unsafe_allow_html=True)


def _sub(text: str) -> None:
    st.markdown(f'<p class="mag-sub">{text}</p>', unsafe_allow_html=True)


def _page_trends() -> None:
    st.title("Trends")
    # The subject is chosen before the description so the description can name it.
    active = _active_domain()
    _lead(
        "You can use this to inspect and understand the latest sentiment in your "
        "field of research and why is it so."
    )
    _sub(f"Never miss an update on the latest scoop of {active.value}.")
    with st.expander("Read the numbers carefully"):
        st.markdown(
            "- **Small batches lie.** A handful of new papers can make any word look "
            "explosive. Where that happens the line says `LOW SUPPORT` — believe the "
            "warning, not the percentage.\n"
            "- **A rising word is not a rising field.** It may be one team's habit, "
            "or a rename of something old.\n"
            "- **It only knows what you collected.** Nothing here is a statement "
            "about the literature at large."
        )
    st.divider()
    _render_banner(active)
    st.divider()
    # Sentiment panel (Spec 11) omitted: the sentiment module is optional and
    # not enabled, and must never be blended into the Topic-Trend Banner.
    _render_frontier(active)


def _page_graph() -> None:
    st.title("Evidence Graph")
    _lead(
        "Use our PageRank system to find out the history, evolution and the world of "
        "theories you are interested."
    )
    _sub(
        "Gain inspiration, a bird's eye view and communicate the lost links between "
        "your theory's origin and where you are right now."
    )
    st.markdown(
        "There is an influence ranking that also determines which paper has triggered "
        "further research on a particular TOI. It's determined by the PageRank "
        "system.\n\n"
        "You may also use a tracer to narrow your lineage scope, if interested to "
        "do so."
    )
    with st.expander("What the map is not"):
        st.markdown(
            "1. The papers are only relevant within the scope of your query and "
            "influence.\n"
            "2. Oldest paper is the oldest from the query, not the oldest itself.\n"
            "3. The lineages aren't contextually accounted for completely, this is "
            "a work in progress."
        )
    st.divider()
    _render_evidence_graph()


def _pick_graph(key: str) -> tuple[int, dict[str, Any]] | None:
    """Shared graph picker: the graph page and the verdict layer both need one."""
    graphs = list_graphs()
    if not graphs:
        st.info(
            "No graphs yet — build one on the **Evidence Graph** page, or run "
            '`magnetor harvest "<your question>"`.'
        )
        return None
    labels = [f"{g['query']}  ·  {g['n_nodes']} papers" for g in graphs]
    choice = st.selectbox(
        "Harvested question", range(len(graphs)), format_func=lambda i: labels[i], key=key
    )
    document = load_graph(str(graphs[choice]["query"]))
    if not document:
        st.warning("Could not load that graph.")
        return None
    return choice, document


def _render_evidence_graph() -> None:
    """Branch C (ADR-0006): the query-relative influence node map."""
    _render_harvest_launcher()
    picked = _pick_graph("graph_page")
    if picked is None:
        return
    choice, document = picked

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
    _render_facets(all_nodes_of(document))
    _render_gaps(document, all_nodes_of(document))

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


def all_nodes_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    raw = document.get("nodes")
    listed = raw if isinstance(raw, list) else []
    return [n for n in listed if isinstance(n, dict)]


def _render_facets(all_nodes: list[dict[str, Any]]) -> None:
    """How this graph's papers approach their subject, and how much is unreadable."""
    counts = facet_counts(all_nodes)
    if not counts or set(counts) == {UNCLASSIFIED}:
        if all_nodes and "facets" not in all_nodes[0]:
            st.caption(
                "Approach facets are not stored in this graph — re-harvest the "
                "question to classify it."
            )
        return
    named = {k: v for k, v in counts.items() if k != UNCLASSIFIED}
    if named:
        spread = " · ".join(f"**{facet}** {count}" for facet, count in named.items())
        st.caption(f"Approaches represented — {spread}.")
    unclassified, blind = unclassified_breakdown(all_nodes)
    if unclassified:
        st.caption(
            f"{unclassified} paper(s) unclassified, {blind} of them because OpenAlex "
            "holds no abstract to read. Facets are a screening heuristic and have "
            "not been checked against human coders."
        )


#: Fields offered in the coding form. The full Part III schema is wider, but the
#: rest needs full text; offering a box that cannot be answered from what is on
#: screen invites guessing, which is the one thing a reliability study cannot
#: survive.
_CODING_FIELDS = (
    "explanation_type",
    "level_sense",
    "interp_schema_type",
    "data_or_phenomenon",
    "normative_subscript",
    "claim_type",
    "core_or_belt",
    "integrity_status",
    # III.3 — the three the II.9 gate runs on, plus the evidential defeaters.
    "independently_measurable",
    "excess_content",
    "use_novel",
    "confirmation_independence",
    "replication_attempt_status",
)


#: How a verdict's standing is shown. An unvalidated or unreliable verdict is
#: demoted visually, not merely footnoted: the gate is deterministic and therefore
#: reads more confident than the coding it rests on.
_STANDING_STYLE = {
    UNVALIDATED: ("⚪", "unvalidated — agreement never measured"),
    UNRELIABLE: ("🔴", "unreliable — coders do not agree"),
    USABLE: ("🟡", "usable — clears the floor, not the firm bar"),
    FIRM: ("🟢", "firm"),
}


def _render_verdict(title: str, qualified: QualifiedVerdict) -> None:
    """Show a result in plain words, then how much weight it carries, then what to do.

    The previous version stacked four internal caveats into one paragraph — what
    the unit of analysis was, what had and had not been measured, what the source
    document calls things. None of it told the reader what to do next, which is
    the only reason to show a caveat at all.
    """
    verdict = qualified.verdict.verdict
    headline, meaning = VERDICT_PLAIN.get(verdict, (verdict, ""))
    mark, _label = _STANDING_STYLE.get(qualified.standing, ("⚪", qualified.standing))
    trust, trust_meaning, what_to_do = STANDING_PLAIN.get(
        qualified.standing, ("", "", "")
    )

    st.markdown(f"#### {title}: {headline}")
    st.markdown(meaning)
    st.caption(f"Called *{verdict}* in the literature. {qualified.verdict.reason}")

    st.markdown(f"**{mark} How much this counts: {trust}** — {trust_meaning}")
    if qualified.is_trustworthy:
        st.caption(what_to_do)
    else:
        st.info(f"**What to do:** {what_to_do}")


def _pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _code_field(name: str, default: str, prefix: str) -> str:
    """One coding control, carrying its own question and its value meanings.

    The question goes under the control where it cannot be missed; where to look
    and what each value asserts go in the tooltip, so a coder who knows the field
    is not slowed by a paragraph they have already read.
    """
    guide = FIELD_GUIDE.get(name)
    options = (UNSET, *FIELDS[name])
    tip = None
    if guide is not None:
        meanings = "\n".join(f"- **{v}** — {m}" for v, m in guide.values.items())
        tip = (
            f"**Where to look.** {guide.look_for}\n\n"
            f"**Values.**\n{meanings}\n\n"
            f"**Effect.** {guide.effect}\n\n"
            f"_Recorded as `{name}`._"
        )
    value = st.selectbox(
        _pretty(name),
        options,
        index=options.index(default) if default in options else 0,
        key=f"{prefix}_{name}",
        help=tip,
    )
    if guide is not None:
        st.caption(guide.question)
    return str(value)


_LEAD_CSS = """
<style>
.mag-lead { font-size: 1.28rem; line-height: 1.55; font-weight: 700; margin: 0 0 .6rem; }
.mag-sub  { font-size: 1.02rem; line-height: 1.6; margin: 0 0 .5rem; }
</style>
"""


def _render_verdict_intro() -> None:
    """Say what the page is for before saying how it works.

    Written for someone who has never met Lakatos. The framework's own words —
    appraisal, gate, defeater, corroboration — describe the machinery, and a
    reader who does not already know the machinery cannot get to the purpose
    through them. Purpose first, mechanism second, vocabulary only where it earns
    its place.
    """
    st.markdown(_LEAD_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="mag-lead">Imre Lakatos was a philosopher of science who had a '
        "bright idea, one fine morning, to test the utility and the chances of "
        "progression of any scientific theory involved; To check its ability to "
        "predict novel predictions and entire new fields in order to further solve "
        "it.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mag-sub">We modeled it mathematically, packaging it in a simple '
        "3 step solution: Choose the paper, set and answer the parameters and yield "
        "the result.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("**The test is three questions about the paper's new idea:**")
    st.markdown(
        "1. **Can you check it separately?** Is there some other measurement or "
        "observation that could confirm what was added — or does it only ever show "
        "up in the very problem it was invented for?\n"
        "2. **Does it reach further?** Does the patched idea say anything about a "
        "situation beyond the problem it was built to solve?\n"
        "3. **Did it call the shot first?** Was that further prediction made before "
        "anyone knew how it would turn out?"
    )
    st.markdown(
        "All three yes → **moving forward**. Fail the first → **falling back**. "
        "Fail either of the others → **standing still**."
    )

    with st.expander("Why three questions can say anything about promise"):
        st.markdown(
            "Because they measure **reach, not correctness**.\n\n"
            "An idea can be wrong and still be moving forward — it was making real "
            "bets that could have failed. An idea can be perfectly correct and "
            "standing still — it stopped making them. What the three questions "
            "measure is how much room is left to explore, which is the thing you "
            "actually need when deciding where to spend your reading time.\n\n"
            "This is why *falling back* is the harshest result. A patch you can only "
            "see inside the problem it was built for cannot fail, and something that "
            "cannot fail cannot teach you anything new."
        )

    with st.expander("Read this before you start"):
        st.markdown(
            "- **It judges your reading, not the paper.** The answers are yours. "
            "Careless answers give a confident-looking result that means nothing.\n"
            "- **One person's answers are an opinion.** Two people answering "
            "separately is the point at which it becomes something you can measure. "
            "The page tells you which of the two you are looking at.\n"
            "- **You need the actual paper** — the methods and results, not the "
            "abstract. Nothing here is read automatically.\n"
            "- **It says nothing about true or false.** An idea can be moving "
            "forward and turn out wrong.\n"
            "- **A missing answer is never a good sign.** Unanswered stays "
            "*undetermined*; it never quietly counts as a pass."
        )


def _page_verdict() -> None:
    """The appraisal workspace: input, settings, output — in that order."""
    st.title("Verdict layer")
    _render_verdict_intro()
    st.divider()
    picked = _pick_graph("verdict_graph")
    if picked is None:
        return
    _choice, document = picked
    all_nodes = all_nodes_of(document)
    if not all_nodes:
        st.info("This graph has no papers to appraise.")
        return
    query = str(document.get("query", ""))

    # ---- 1. what goes in -------------------------------------------------
    st.header("Step 1 · Choose the paper")
    st.markdown(
        "**What you do:** pick a paper you have actually read.\n\n"
        "**What this does:** decides which idea gets judged. Nothing is read for "
        "you — this tool only knows titles, dates and who cited whom, never what "
        "any paper says. So the whole judgement rests on your reading.\n\n"
        "**Why it comes first:** the three questions in Step 2 can only be answered "
        "by someone who has seen the paper's methods and results. Expect five to "
        "ten minutes per paper."
    )
    labels = [
        f"{i + 1}. {_short(n, str(n.get('id')), 70)}" for i, n in enumerate(all_nodes)
    ]
    index = st.selectbox(
        "Paper to appraise", range(len(all_nodes)), format_func=lambda i: labels[i],
        key="verdict_paper",
    )
    node = all_nodes[index]
    paper_id = str(node.get("id"))
    url = str(node.get("url") or "") or None
    st.markdown(f"### {linked(str(node.get('title') or paper_id), url)}")
    st.caption(
        f"{node.get('year') or 'year unknown'} · {node.get('venue') or 'venue unknown'} · "
        f"influence {float(node.get('influence') or 0):.2f} · "
        f"in-set citations {node.get('in_degree', 0)}"
    )
    st.caption(
        f"Your answers will be saved under the name **{current_coder()}** — set "
        "`MAGNETOR_CODER` to change it. Names matter because the whole point is "
        "comparing two readers."
    )

    # ---- 2. what each setting does --------------------------------------
    st.header("Step 2 · Answer the questions")
    st.markdown(
        "**What you do:** answer three questions about the paper from Step 1. Then, "
        "if you can, four more about how much the evidence behind it can be "
        "trusted.\n\n"
        "**What this does:** your three answers — and only those three — decide the "
        "result in Step 3. The four extra ones can *cancel* a good result but can "
        "never create one, because trustworthy evidence for a patch that goes "
        "nowhere is still a patch that goes nowhere."
    )
    defaults = prefill(node)
    chosen: dict[str, str] = {}

    st.subheader("The three that decide")
    st.markdown(
        "All three must be *yes* for **moving forward**. A confident yes here "
        "cannot make up for a no there — each one has to hold on its own."
    )
    for name in fields_by_role(GATE):
        chosen[name] = _code_field(name, defaults.get(name, UNSET), "verdict")

    st.subheader("Reasons to distrust a good result")
    st.markdown(
        "These ask *who checked the work*. If a paper passes the three questions "
        "but nobody outside the original group ever tested it, the result still "
        "stands — you just should not lean on it. Answering these cancels a pass "
        "outright rather than nudging it down, so the reason is always visible "
        "instead of hidden inside a number."
    )
    for name in fields_by_role(DEFEATER):
        chosen[name] = _code_field(name, defaults.get(name, UNSET), "verdict")

    with st.expander("Optional · describe the paper (changes nothing above)"):
        st.markdown(
            "These record *what kind* of claim the paper makes. They do not touch "
            "the result. They exist so that a collection of judged papers can later "
            "be searched and compared — and because if two readers cannot agree on "
            "them, that tells you the framework's own categories are unclear."
        )
        for name in fields_by_role(CONTEXT):
            chosen[name] = _code_field(name, defaults.get(name, UNSET), "verdict")

    note = st.text_input(
        "Anything you want to remember about this judgement (optional)",
        key="verdict_note",
    )

    # ---- 3. what comes out ----------------------------------------------
    st.header("Step 3 · Read the result")
    st.markdown(
        "**What you do:** read the two lines below, then save if you are happy with "
        "your answers.\n\n"
        "**What this does:** turns your Step 2 answers into a judgement about where "
        "this line of work is heading — and tells you how much that judgement is "
        "worth, which depends entirely on whether anyone else has answered too.\n\n"
        "**Why it needs the other steps:** the first line is only ever as careful as "
        "your reading in Step 1. The second line only becomes meaningful once a "
        "second person has done Step 2 without seeing your answers."
    )
    claims = load_coding(query)
    live = qualify(judge(chosen), claims)
    _render_verdict("This paper", live)
    if live.verdict.is_defeated:
        st.warning(
            f"**But do not lean on it:** {live.verdict.defeated_by}. The paper passed "
            "the three questions; the evidence underneath it will not carry the weight."
        )

    if st.button("Save this judgement", key="verdict_save", type="primary"):
        if all(value == UNSET for value in chosen.values()):
            st.warning("Nothing answered yet — answer at least one question first.")
        else:
            record_coding(query, paper_id, chosen, note=note)
            st.success(
                "Saved. Earlier answers are never overwritten, so if someone else "
                "judges this paper differently, both are kept — that difference is "
                "what makes the second line meaningful."
            )

    if not claims:
        st.info(
            "**Nothing saved for this question yet.** Save your answers, then have "
            "someone else judge the same paper without seeing them. That is the step "
            "that turns an opinion into something you can measure."
        )
        return

    st.divider()
    st.subheader("All the papers judged for this question")
    st.markdown(
        "One paper tells you about one idea. Several tell you whether the whole line "
        "of work is still moving — including whether it stopped years ago."
    )
    years = {
        str(n.get("id")): (n.get("year") if isinstance(n.get("year"), int) else None)
        for n in all_nodes
    }
    rolled = qualify(programme_verdict(claims, years=years), claims)
    _render_verdict("This line of work", rolled)
    st.caption(
        f"A line of work counts as stalled if its most recent forward step is more "
        f"than {DEFAULT_DEGENERATION_WINDOW} years old. That number is a choice we "
        "made, not something measured — change it if your field moves faster."
    )

    with st.expander(f"Where people agreed and disagreed ({len(claims)} judgements)"):
        st.markdown(
            "Each row is one question. A high number means independent readers gave "
            "the same answer; a low one means the question is being read differently "
            "by different people, which is worth knowing before you trust any result "
            "built on it."
        )
        for item in reliability_report(claims):
            agreement = (
                "not measurable yet" if item.alpha is None else f"{item.alpha:.2f}"
            )
            st.markdown(
                f"- **{_pretty(item.field_name)}** — agreement {agreement} "
                f"· {item.units} paper(s) judged twice by {item.coders} reader(s)"
            )
        st.caption(
            f"Above {ALPHA_FLOOR} is usable, above {ALPHA_FIRM} is solid. Those two "
            "cut-offs are conventions from the research-methods literature, not "
            "measurements of anything here."
        )
        st.download_button(
            "Download all judgements (CSV)",
            data=to_csv(claims),
            file_name=f"{slugify(query)}-judgements.csv",
            mime="text/csv",
            key="verdict_download",
        )


def _render_gaps(document: dict[str, Any], all_nodes: list[dict[str, Any]]) -> None:
    """What this map is missing: works it leans on, and axes nobody took."""
    found = read_gaps(document)
    axes = facet_gaps(all_nodes)
    if not found and not axes:
        return
    with st.expander("Open gaps — what this map is missing"):
        if found:
            st.markdown(
                "**Foundational works cited from inside this set but absent from it.** "
                "Boundary leakage says how much points outside; this says what."
            )
            for gap in found[:10]:
                label = gap.title or f"`{gap.openalex_id}` (no metadata returned)"
                st.markdown(
                    f"- [{label}]({gap.url}) — cited by **{gap.cited_by_in_set}** "
                    f"paper(s) here ({gap.share:.0%})"
                    + (f" · {gap.year}" if gap.year else "")
                )
            st.caption(
                "Some entries resolve to no metadata — usually a merged or withdrawn "
                "OpenAlex record. They keep their count rather than being dropped, "
                "because removing them would understate the incompleteness."
            )
        if axes:
            absent = [g.facet for g in axes if g.absent]
            thin = [(g.facet, g.papers) for g in axes if not g.absent]
            st.markdown("**Approaches barely taken on this question.**")
            if absent:
                st.markdown(f"- Not represented at all: {', '.join(absent)}")
            for facet, papers in thin:
                st.markdown(f"- `{facet}` — only {papers} paper(s)")
        st.caption(
            "A gap is an absence in a harvested slice, and absence has cheap "
            "explanations: not indexed, not applicable, or tried and uncited. Read "
            "this as a prompt for attention, never as a verdict about the field."
        )


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
    st.markdown("**For narrower questions**")
    st.caption(
        "The density referred to is how much each paper is interconnected to each "
        "other."
    )
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
        st.caption("To appraise this paper's revision, use the **Verdict layer** page.")


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
    facets = node_facets(by_id.get(node_id, {}))
    if facets != (UNCLASSIFIED,):
        st.markdown("**Approach:** " + " · ".join(facets))
    evidence = (by_id.get(node_id) or {}).get("facet_evidence")
    if isinstance(evidence, dict) and evidence:
        terms = "; ".join(f"{k}: {', '.join(v)}" for k, v in evidence.items())
        st.caption(f"Classified from these terms — {terms}")

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

        cross = cross_facet_neighbours(node_id, list(related), by_id)
        if cross:
            st.markdown("**...of those, approaching it differently:**")
            for other_id, weight, different in cross:
                other = by_id.get(other_id)
                title = linked(
                    str((other or {}).get("title") or other_id),
                    str((other or {}).get("url") or "") or None,
                )
                st.markdown(
                    f"- {title} <small>— **{', '.join(different)}** (weight {weight})</small>",
                    unsafe_allow_html=True,
                )


# Streamlit runs the entry script in a module named "__main__" and re-executes
# it top-to-bottom on every interaction, so calling main() here re-renders each
# rerun. Guarded so importing this module (e.g. from streamlit_app.py) does NOT
# double-run — that entry calls main() itself.
if __name__ == "__main__":
    main()
