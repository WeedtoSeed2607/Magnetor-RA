"""View-model helpers for the dashboard (Spec Section 11).

Pure, Streamlit-free functions so the panel content is unit-testable. The
Streamlit script (``dashboard.py``) imports these and only handles rendering.

Panel mapping (Spec 11):
- Topic-Trend Banner  <- Branch A persisted trends (statistic-anchored lines)
- Frontier Feed       <- Branch A anomalies, hot clusters, recent papers
- Primary Viewport    <- Branch B deep-dive (built live in the app)
- Sentiment panel     <- optional module (deferred), never blended into the banner
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from magnetor.citations import Citation
from magnetor.config import get_domain_config
from magnetor.resources import DomainStore
from magnetor.trends import TRENDS_FILENAME
from magnetor.types import Domain, Paper


@dataclass(frozen=True, slots=True)
class FrontierFeed:
    """Branch-A secondary panel content (Spec 11)."""

    anomalies: tuple[dict[str, object], ...]  # emerging terms, volume-normalized
    hot_clusters: tuple[dict[str, object], ...]  # topics by latest-slice prevalence
    recent: tuple[Paper, ...]  # newest stored papers


def load_trends(domain: Domain) -> dict[str, object] | None:
    """Read a domain's persisted ``trends.json``, or ``None`` if not run yet."""
    path = get_domain_config(domain).storage_dir / TRENDS_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def banner_lines(trends: dict[str, object] | None) -> list[str]:
    """Statistic-anchored Topic-Trend Banner text (Spec 11) — no narrative-mood."""
    if not trends:
        return []
    interpretation = trends.get("interpretation")
    return [str(line) for line in interpretation] if isinstance(interpretation, list) else []


def frontier_feed(
    domain: Domain, trends: dict[str, object] | None, *, recent: int = 5
) -> FrontierFeed:
    """Assemble the Frontier Feed: anomalies, hot clusters, recent papers."""
    store = DomainStore(domain, get_domain_config(domain).storage_dir)
    recents = tuple(store.read_records(limit=recent))
    anomalies = _as_dicts(trends.get("anomalies") if trends else None)
    hot = _hot_clusters(trends)
    return FrontierFeed(anomalies=anomalies, hot_clusters=hot, recent=recents)


def _hot_clusters(trends: dict[str, object] | None) -> tuple[dict[str, object], ...]:
    if not trends:
        return ()
    topics = trends.get("topics")
    if not isinstance(topics, list):
        return ()
    dicts = [t for t in topics if isinstance(t, dict)]
    return tuple(sorted(dicts, key=_latest_prevalence, reverse=True))


def _latest_prevalence(topic: dict[str, object]) -> float:
    prevalence = topic.get("prevalence")
    if isinstance(prevalence, list) and prevalence:
        last = prevalence[-1]
        return float(last) if isinstance(last, (int, float)) else 0.0
    return 0.0


def _as_dicts(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _arxiv_abs(external_id: str) -> str:
    return f"https://arxiv.org/abs/{external_id}"


def paper_url(paper: Paper) -> str | None:
    """Best canonical landing page for a stored paper, or ``None`` if unknown.

    arXiv → abstract page; PubMed Central → PMC article; otherwise DOI, then
    whatever ``pdf_url`` the source recorded. Source-specific because arXiv and
    PubMed use different identifier schemes.
    """
    source = (paper.source or "").lower()
    ext = paper.external_id or ""
    if "arxiv" in source and ext:
        return _arxiv_abs(ext)
    if ext.upper().startswith("PMC"):
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{ext}/"
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    return paper.pdf_url


def citation_url(citation: Citation) -> str | None:
    """Resolve a citation neighbour to a URL: DOI first, then arXiv id."""
    if citation.doi:
        return f"https://doi.org/{citation.doi}"
    if citation.arxiv_id:
        return _arxiv_abs(citation.arxiv_id)
    return None


def linked(title: str, url: str | None) -> str:
    """A Markdown link when a URL exists, else the plain title."""
    return f"[{title}]({url})" if url else title


def _dot_escape(text: str) -> str:
    return text.replace("\\", " ").replace('"', "'").replace("\n", " ")


def _short_label(title: str, words: int = 4) -> str:
    parts = title.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


#: L5.1(3) — mandatory honesty cue beside the graph. Influence is percentile-
#: normalised *within the harvested set*, so it is a rough guide, not a verdict.
GRAPH_DISCLAIMER = (
    "Influence levels are a **rough guide only**. They are percentile-ranked "
    "within the papers this harvest captured — not absolute or authoritative "
    "measures — and a traced pathway is a ranked suggestion, not established "
    "history."
)

#: Traced-leg colours. Distinct from the node fills (red/orange/blue) so a leg
#: cannot be confused with an integrity or stability flag. ``link`` marks a
#: checked connection between hand-picked nodes rather than a traced progression.
_LEG_COLOURS = {"roots": "#8e44ad", "development": "#1f9d55", "link": "#0b7285"}

#: Derived relation layers (ADR-0006 L4). Drawn dashed and arrowless so they are
#: structurally unmistakable against the solid, arrowed citation backbone — I2
#: forbids mixing derived edges into it. Muted on purpose: they are navigational.
_RELATION_STYLES = {
    "biblio_coupled": ("#8d6e63", "dashed"),  # shared ancestry (Kessler)
    "co_cited": ("#5c6bc0", "dotted"),  # cited together (Small)
}

RELATION_LABELS = {
    "biblio_coupled": "shares references with",
    "co_cited": "cited alongside",
}


def relation_rows(document: Mapping[str, object], kind: str) -> tuple[tuple[str, str, int], ...]:
    """Parse one derived relation layer, tolerating graphs harvested before it existed."""
    raw = document.get(kind)
    if not isinstance(raw, list):
        return ()
    rows: list[tuple[str, str, int]] = []
    for row in raw:
        if isinstance(row, list) and len(row) == 3:
            weight = row[2]
            rows.append((str(row[0]), str(row[1]), int(weight) if isinstance(weight, int) else 0))
    return tuple(rows)


def related_papers(
    document: Mapping[str, object], node_id: str, kind: str, *, limit: int = 6
) -> tuple[tuple[str, int], ...]:
    """Strongest derived neighbours of one paper, heaviest first.

    This is the answer to "what leans on the same foundations as this, without
    either citing the other" — the relation the citation backbone cannot express.
    """
    found = [
        (b if a == node_id else a, weight)
        for a, b, weight in relation_rows(document, kind)
        if node_id in (a, b)
    ]
    found.sort(key=lambda item: (-item[1], item[0]))
    return tuple(found[:limit])


def graph_dot(
    document: dict[str, object],
    *,
    top_n: int = 40,
    traced: Mapping[tuple[str, str], str] | None = None,
    highlighted: Collection[str] | None = None,
    anchor: str | None = None,
    layers: Collection[str] = (),
    only_nodes: Collection[str] | None = None,
) -> str:
    """Render an Evidence Graph document (ADR-0006 L4) as a Graphviz DOT string.

    Node size = influence; fill encodes state — red retracted (integrity), orange
    unstable rank (bootstrap CI), blue otherwise. Streamlit renders DOT
    client-side, so no system Graphviz binary is required.

    ``traced`` maps a stored ``(citing, cited)`` edge to its leg (``roots`` /
    ``development`` / ``link``); those edges are drawn thick and leg-coloured so a
    pathway can be followed by eye (L5.1(2), L5.3). ``highlighted`` nodes get a
    bold outline and ``anchor`` a doubled one. Nodes appearing in a trace or
    highlight are drawn **even when they rank below** ``top_n`` — otherwise a
    traced step could point at a node that was never emitted, silently breaking
    the path.

    ``layers`` names derived relation layers to overlay (``biblio_coupled``,
    ``co_cited``); they render dashed and arrowless, never merged into the
    citation backbone (I2). ``only_nodes`` restricts the drawing to a subgraph,
    which is how a cluttered graph is reduced to just the papers under inspection.
    """
    traced = traced or {}
    highlighted = set(highlighted or ())
    all_nodes = list(_as_dicts(document.get("nodes")))
    pinned = {n for edge in traced for n in edge} | highlighted
    if anchor:
        pinned.add(anchor)
    nodes = [n for n in all_nodes[:top_n]]
    shown = {str(n.get("id")) for n in nodes}
    nodes += [n for n in all_nodes[top_n:] if str(n.get("id")) in pinned - shown]
    if only_nodes is not None:
        wanted = set(only_nodes)
        nodes = [n for n in nodes if str(n.get("id")) in wanted]
    keep = {str(n.get("id")) for n in nodes}

    lines = [
        "digraph EvidenceGraph {",
        "  rankdir=LR;",
        '  bgcolor="transparent";',
        '  node [shape=circle, style=filled, fixedsize=true, fontsize=8,'
        ' fontname="sans-serif"];',
    ]
    for node in nodes:
        influence = _as_float(node.get("influence"))
        width = 0.4 + 1.6 * influence
        if node.get("is_retracted"):
            fill = "#d9534f"  # retracted — integrity flag
        elif node.get("stable") is False:
            fill = "#f0ad4e"  # rank unstable under bootstrap
        else:
            fill = "#5b9bd5"
        label = _dot_escape(_short_label(str(node.get("title") or node.get("id"))))
        nid = _dot_escape(str(node.get("id")))
        attrs = [f'label="{label}"', f"width={width:.2f}", f'fillcolor="{fill}"']
        if nid == anchor:
            attrs += ['color="#111827"', "penwidth=3.0", "peripheries=2"]
        elif nid in highlighted:
            attrs += ['color="#111827"', "penwidth=2.5"]
        lines.append(f'  "{nid}" [{", ".join(attrs)}];')

    edges = document.get("edges")
    for edge in edges if isinstance(edges, list) else []:
        if isinstance(edge, list) and len(edge) == 2:
            u, v = str(edge[0]), str(edge[1])
            if u not in keep or v not in keep:
                continue
            leg = traced.get((u, v))
            if leg:
                colour = _LEG_COLOURS.get(leg, "#111827")
                lines.append(
                    f'  "{u}" -> "{v}" [color="{colour}", penwidth=3.0, arrowsize=0.9];'
                )
            else:
                # Darker than the original near-white grey (L5.1(2)) so untraced
                # citations stay legible without competing with a traced leg.
                lines.append(f'  "{u}" -> "{v}" [color="#8a8a8a", arrowsize=0.5];')

    for layer in layers:
        colour, style = _RELATION_STYLES.get(layer, ("#999999", "dashed"))
        for a, b, weight in relation_rows(document, layer):
            if a in keep and b in keep:
                # dir=none: a derived relation is symmetric and asserts no
                # precedence, so drawing an arrowhead would misstate it.
                lines.append(
                    f'  "{a}" -> "{b}" [color="{colour}", style={style}, dir=none, '
                    f'penwidth=0.9, constraint=false, tooltip="{weight}"];'
                )
    lines.append("}")
    return "\n".join(lines)


def node_detail(node: Mapping[str, object]) -> tuple[str, ...]:
    """Per-metric breakdown lines for the node detail panel (L5.1(1), C1).

    Deliberately no abstract: a graph artifact stores identifiers, metrics and
    edges only (§3/I4), so the panel gives identity, the link out to the paper,
    and every score component that produced the node's size — never a body.
    Absent values render as "unavailable" rather than 0 (C3).
    """
    lines = [
        f"**Year:** {node.get('year') or 'unavailable'}",
        f"**Venue:** {node.get('venue') or 'unavailable'}",
        f"**Influence (PageRank percentile):** {_as_float(node.get('influence')):.3f}",
        f"**In-set citations (in-degree):** {node.get('in_degree', 0)}",
        f"**PageRank (raw):** {_as_float(node.get('pagerank')):.6f}",
    ]
    if node.get("median_rank") is not None:
        stability = "stable" if node.get("stable") else "unstable — treat with caution"
        lines.append(
            f"**Bootstrap rank:** {node.get('median_rank')} "
            f"(95% CI {node.get('lo_rank')}-{node.get('hi_rank')}) · {stability}"
        )
    else:
        lines.append("**Bootstrap rank:** unavailable")
    if node.get("is_retracted"):
        lines.append("🔴 **Retracted** — recorded from OpenAlex.")
    if node.get("is_review"):
        lines.append("📄 Review article — its reference list reflects its authors' emphases.")
    return tuple(lines)


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def search_access(
    expected: str | None, entered: str | None, already_ok: bool
) -> tuple[bool, str | None]:
    """Gate the live deep-dive (which spends API quota) behind a password.

    Returns ``(unlocked, error_message)``. When no password is configured
    (``expected`` falsy), the search is open — that's the local-dev default.
    Once unlocked in a session (``already_ok``) it stays unlocked.
    """
    if not expected or already_ok:
        return True, None
    if entered and entered == expected:
        return True, None
    if entered:
        return False, "Incorrect password"
    return False, None
