"""Branch B Technical Deep-Dive path selection (Spec Section 7.2).

Given the retrieval hits from the router (Phase 2), decide between the two paths
the spec defines:

- **Anchor-Lock (Path A):** top score >= the domain's calibrated threshold ->
  lock the single best paper as source of truth and expand its forward/backward
  citations via Semantic Scholar.
- **Field-Map (Path B / Fallback):** top score below threshold -> map the top
  3-5 competing positions across the retrieved abstracts.

The result is deliberately LLM-free to *produce* but LLM-ready to *consume*:
:func:`render_grounded_context` emits a payload whose factual sections (domain
labels, paper identities, abstracts, scores, citations) are explicitly marked
GROUNDED and immutable, so a downstream Critic / synthesis model (Claude
Sonnet / Opus / Haiku) can be fed and tested against it without altering ground
truth — the seam the spec's Sections 8-9 plug into.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from magnetor.citations import Citation
from magnetor.config import get_domain_config
from magnetor.embeddings.base import Embedder
from magnetor.resources import DomainStore
from magnetor.router import CrossDomainRouter, Routing, retrieve
from magnetor.types import Domain, Paper

#: Field-Map surfaces at most this many competing positions (Spec 7.2: "3-5").
DEFAULT_POSITIONS = 5


class Path(StrEnum):
    ANCHOR_LOCK = "anchor_lock"
    FIELD_MAP = "field_map"


@runtime_checkable
class CitationExpander(Protocol):
    """Boundary for citation expansion (Semantic Scholar in production)."""

    def expand(self, paper: Paper) -> tuple[list[Citation], list[Citation]]:
        ...


@dataclass(frozen=True, slots=True)
class AnchorLock:
    """The locked source-of-truth paper plus its citation neighbourhood."""

    paper: Paper
    score: float
    forward: tuple[Citation, ...]  # papers that cite the anchor
    backward: tuple[Citation, ...]  # papers the anchor cites


@dataclass(frozen=True, slots=True)
class Position:
    """One competing position in a Field-Map — a real retrieved abstract."""

    rank: int
    paper: Paper
    score: float


@dataclass(frozen=True, slots=True)
class FieldMap:
    positions: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class DeepDiveResult:
    """Outcome of Branch B path selection for one query."""

    query: str
    domain: Domain | None
    path: Path | None  # None when nothing was retrieved
    top_score: float | None
    threshold: float | None
    anchor: AnchorLock | None = None
    field_map: FieldMap | None = None
    routing: Routing | None = None


def build_deep_dive(
    embedder: Embedder,
    router: CrossDomainRouter,
    query: str,
    expander: CitationExpander,
    *,
    k: int = DEFAULT_POSITIONS,
    domain: Domain | None = None,
    threshold: float | None = None,
) -> DeepDiveResult:
    """Route + retrieve, then select Anchor-Lock or Field-Map per Spec 7.2.

    ``threshold`` overrides the domain's configured Anchor-Lock threshold for
    this run — the operator-facing knob until per-domain calibration exists.
    """
    routing, hits = retrieve(embedder, router, query, k=k, domain=domain)
    if not hits:
        return DeepDiveResult(
            query=query, domain=None, path=None, top_score=None,
            threshold=None, routing=routing,
        )

    top = hits[0]
    config = get_domain_config(top.domain)
    threshold = config.anchor_threshold if threshold is None else threshold
    papers = _load_papers({hit.domain for hit in hits})

    if top.score >= threshold:
        anchor_paper = papers.get((top.domain, top.external_id))
        forward, backward = expander.expand(anchor_paper) if anchor_paper else ([], [])
        anchor = None
        if anchor_paper is not None:
            anchor = AnchorLock(
                paper=anchor_paper,
                score=top.score,
                forward=tuple(forward),
                backward=tuple(backward),
            )
        return DeepDiveResult(
            query=query, domain=top.domain, path=Path.ANCHOR_LOCK,
            top_score=top.score, threshold=threshold, anchor=anchor, routing=routing,
        )

    positions = [
        Position(rank=rank, paper=papers[(hit.domain, hit.external_id)], score=hit.score)
        for rank, hit in enumerate(hits[:DEFAULT_POSITIONS], start=1)
        if (hit.domain, hit.external_id) in papers
    ]
    return DeepDiveResult(
        query=query, domain=top.domain, path=Path.FIELD_MAP,
        top_score=top.score, threshold=threshold,
        field_map=FieldMap(tuple(positions)), routing=routing,
    )


def _load_papers(domains: set[Domain]) -> dict[tuple[Domain, str], Paper]:
    papers: dict[tuple[Domain, str], Paper] = {}
    for domain in domains:
        store = DomainStore(domain, get_domain_config(domain).storage_dir)
        for paper in store.read_records():
            papers[(domain, paper.external_id)] = paper
    return papers


_GROUNDED_HEADER = (
    "# MAGNETOR GROUNDED CONTEXT - Branch B Technical Deep-Dive (Spec Section 7.2)\n"
    "#\n"
    "# Sections marked [GROUNDED] are source-of-truth: domain labels, paper\n"
    "# identities, titles, abstracts, scores, and citations are verbatim from the\n"
    "# retrieval and Semantic Scholar layers. A downstream model (Claude Sonnet /\n"
    "# Opus / Haiku) MUST NOT alter, paraphrase, invent, or contradict anything\n"
    "# marked [GROUNDED]; it may only write inside the SYNTHESIS SLOT at the end.\n"
)

_SYNTHESIS_SLOT = (
    "=== SYNTHESIS SLOT (model output only) ===\n"
    "# A model may summarize the material above here. Constraints:\n"
    "#  - Do not add papers, citations, numbers, or domains not present above.\n"
    "#  - Do not change the routed domain or the path decision.\n"
    "#  - Tag each statement Grounded / Illustrative / Contested (Spec Section 9).\n"
)


def render_grounded_context(result: DeepDiveResult) -> str:
    """Render an LLM-ready payload with grounded (immutable) sections marked."""
    lines: list[str] = [_GROUNDED_HEADER, ""]
    lines.append(f"QUERY: {result.query}")
    if result.path is None:
        lines.append("PATH: (no results) [GROUNDED]")
        lines.append("")
        lines.append(_SYNTHESIS_SLOT)
        return "\n".join(lines)

    lines.append(f"ROUTED DOMAIN: {result.domain.value if result.domain else '-'}"
                 "        [GROUNDED - do not change]")
    lines.append(f"PATH: {result.path.value}        [GROUNDED - decided by score vs threshold]")
    lines.append(
        f"TOP SCORE: {result.top_score:.3f}   THRESHOLD: {result.threshold:.3f}   [GROUNDED]"
    )
    lines.append("")

    if result.path is Path.ANCHOR_LOCK and result.anchor is not None:
        lines.extend(_render_anchor(result.anchor))
    elif result.path is Path.FIELD_MAP and result.field_map is not None:
        lines.extend(_render_field_map(result.field_map))

    lines.append("")
    lines.append(_SYNTHESIS_SLOT)
    return "\n".join(lines)


def _render_anchor(anchor: AnchorLock) -> list[str]:
    p = anchor.paper
    out = [
        "--- ANCHOR PAPER [GROUNDED - source of truth - verbatim] ---",
        f"id: {p.external_id}   domain: {p.domain.value}   score: {anchor.score:.3f}",
        f"title: {p.title}",
        f"abstract: {p.abstract or '(none)'}",
        "",
        "--- FORWARD CITATIONS (papers citing the anchor) [GROUNDED] ---",
    ]
    out.extend(_render_citations(anchor.forward))
    out.append("")
    out.append("--- BACKWARD REFERENCES (papers the anchor cites) [GROUNDED] ---")
    out.extend(_render_citations(anchor.backward))
    return out


def _render_field_map(field_map: FieldMap) -> list[str]:
    out = ["--- COMPETING POSITIONS [GROUNDED - each is a real retrieved abstract] ---"]
    if not field_map.positions:
        out.append("(none)")
    for pos in field_map.positions:
        p = pos.paper
        out.append(
            f"[{pos.rank}] domain={p.domain.value}  id={p.external_id}  score={pos.score:.3f}"
        )
        out.append(f"    title: {p.title}")
        out.append(f"    abstract: {p.abstract or '(none)'}")
    return out


def _render_citations(citations: Sequence[Citation]) -> list[str]:
    if not citations:
        return ["(none found)"]
    rendered: list[str] = []
    for c in citations:
        ident = c.doi or (f"arXiv:{c.arxiv_id}" if c.arxiv_id else "")
        year = f" ({c.year})" if c.year else ""
        suffix = f" [{ident}]" if ident else ""
        rendered.append(f"- {c.title}{year}{suffix}")
    return rendered
