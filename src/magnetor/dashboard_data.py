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


def graph_dot(document: dict[str, object], *, top_n: int = 40) -> str:
    """Render an Evidence Graph document (ADR-0006 L4) as a Graphviz DOT string.

    Node size = influence; colour encodes state — red retracted (integrity),
    orange unstable rank (bootstrap CI), blue otherwise. Streamlit renders DOT
    client-side, so no system Graphviz binary is required.
    """
    nodes = list(_as_dicts(document.get("nodes")))[:top_n]
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
        lines.append(f'  "{nid}" [label="{label}", width={width:.2f}, fillcolor="{fill}"];')
    edges = document.get("edges")
    for edge in edges if isinstance(edges, list) else []:
        if isinstance(edge, list) and len(edge) == 2:
            u, v = str(edge[0]), str(edge[1])
            if u in keep and v in keep:
                lines.append(f'  "{u}" -> "{v}" [color="#bbbbbb", arrowsize=0.5];')
    lines.append("}")
    return "\n".join(lines)


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
