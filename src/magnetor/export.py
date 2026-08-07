"""Branch C — packaging a traced lineage as a downloadable citation bundle.

Turns a set of routes through the Evidence Graph into a folder the researcher can
keep: a reading order, a spreadsheet, and a BibTeX file that imports straight
into a reference manager.

**What this deliberately does not contain: the papers themselves.** Most are
paywalled and bulk-retrieving full texts is not ours to do, so every entry
carries a resolvable DOI (or OpenAlex) link instead. The bundle is a
bibliography, not a library.

**Nor does it contain abstracts** — a graph artifact holds identifiers, metrics
and edges only (ADR-0006 section 3 / I4), so there is no body text to export.

The provenance and caveats travel *with* the download rather than living only in
the UI. A file that leaves the app loses its context otherwise, and every number
here is percentile-normalised within one harvest — meaningless once separated
from that fact.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Fixed timestamp so the same lineage always produces a byte-identical archive.
#: Zip entries otherwise embed the current clock, which would make two exports of
#: unchanged data differ.
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

_BIBTEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True, slots=True)
class LineageSet:
    """One traced relationship and every route found for it."""

    label: str
    kind: str  # lineage | indirect | progression
    routes: tuple[tuple[str, ...], ...]


def slugify(text: str, *, fallback: str = "evidence-graph") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or fallback


def _nodes_by_id(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = document.get("nodes")
    listed = raw if isinstance(raw, list) else []
    return {str(n.get("id")): n for n in listed if isinstance(n, dict)}


def _ordered_papers(
    document: Mapping[str, Any], sets: Sequence[LineageSet]
) -> list[dict[str, Any]]:
    """Every paper appearing on any route, in first-encountered reading order."""
    by_id = _nodes_by_id(document)
    seen: dict[str, dict[str, Any]] = {}
    for lineage in sets:
        for route in lineage.routes:
            for node_id in route:
                node = by_id.get(node_id)
                if node is not None and node_id not in seen:
                    seen[node_id] = node
    return list(seen.values())


def _bibtex_escape(text: str) -> str:
    return "".join(_BIBTEX_ESCAPES.get(ch, ch) for ch in text)


def _citation_key(node: Mapping[str, Any]) -> str:
    """A stable, unique key. Author names are not stored on a graph node, so the
    key is built from the title, year and OpenAlex id rather than the usual
    author-year form."""
    title = str(node.get("title") or "")
    word = next((w.lower() for w in re.findall(r"[A-Za-z]{4,}", title)), "paper")
    year = node.get("year") or "n.d."
    return f"{word}{year}_{node.get('id') or 'unknown'!s}"


def papers_bibtex(papers: Sequence[Mapping[str, Any]]) -> str:
    """BibTeX for a reference manager.

    Emitted as ``@article`` with no ``author`` field: the harvest fetches
    authorships but the graph document does not persist them, so inventing a
    name would be worse than omitting one. Add author storage to the node schema
    and this fills in.
    """
    entries: list[str] = []
    for node in papers:
        fields = [f"  title = {{{_bibtex_escape(str(node.get('title') or ''))}}}"]
        if node.get("year") is not None:
            fields.append(f"  year = {{{node['year']}}}")
        if node.get("venue"):
            fields.append(f"  journal = {{{_bibtex_escape(str(node['venue']))}}}")
        if node.get("doi"):
            fields.append(f"  doi = {{{_bibtex_escape(str(node['doi']))}}}")
        if node.get("url"):
            fields.append(f"  url = {{{_bibtex_escape(str(node['url']))}}}")
        note = "Retrieved via OpenAlex; influence is relative to one harvest."
        if node.get("is_retracted"):
            note = "RETRACTED. " + note
        fields.append(f"  note = {{{note}}}")
        entries.append("@article{" + _citation_key(node) + ",\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + ("\n" if entries else "")


def papers_csv(papers: Sequence[Mapping[str, Any]]) -> str:
    """One row per paper, with the score components that produced its size."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "openalex_id", "title", "year", "venue", "doi", "url",
            "influence_pagerank_percentile", "in_set_citations", "pagerank_raw",
            "rank_median", "rank_ci_low", "rank_ci_high", "rank_stable",
            "is_retracted", "is_review",
        ]
    )
    for node in papers:
        writer.writerow(
            [
                node.get("id", ""), node.get("title", ""), node.get("year", ""),
                node.get("venue", ""), node.get("doi", ""), node.get("url", ""),
                node.get("influence", ""), node.get("in_degree", ""),
                node.get("pagerank", ""), node.get("median_rank", ""),
                node.get("lo_rank", ""), node.get("hi_rank", ""),
                node.get("stable", ""), bool(node.get("is_retracted")),
                bool(node.get("is_review")),
            ]
        )
    return buffer.getvalue()


def lineages_markdown(
    document: Mapping[str, Any], sets: Sequence[LineageSet]
) -> str:
    """The routes in reading order, oldest first, with links."""
    by_id = _nodes_by_id(document)
    out = [f"# Lineages — {document.get('query', 'evidence graph')}", ""]
    for lineage in sets:
        out.append(f"## {lineage.label}")
        out.append(f"_Relationship: {lineage.kind}_")
        out.append("")
        if not lineage.routes:
            out.append("No route found within the harvested set.")
            out.append("")
            continue
        for index, route in enumerate(lineage.routes, start=1):
            header = f"### Route {index}" if len(lineage.routes) > 1 else "### Route"
            out.append(f"{header} — {len(route)} papers")
            for step, node_id in enumerate(route, start=1):
                node = by_id.get(node_id, {})
                title = str(node.get("title") or node_id)
                year = node.get("year") or "?"
                url = node.get("url")
                flag = " **[RETRACTED]**" if node.get("is_retracted") else ""
                link = f"[{title}]({url})" if url else title
                out.append(f"{step}. ({year}) {link}{flag}")
            out.append("")
    return "\n".join(out)


def readme(document: Mapping[str, Any], sets: Sequence[LineageSet], n_papers: int) -> str:
    leakage = document.get("boundary_leakage")
    leak_line = (
        f"- **Boundary leakage: {float(leakage):.0%}** — that share of citations from "
        "these papers points *outside* the harvested set, so this is a slice of the "
        "lineage, not the whole of it."
        if isinstance(leakage, (int, float))
        else "- Boundary leakage not recorded for this graph."
    )
    return "\n".join(
        [
            f"# {document.get('query', 'Evidence graph')} — traced lineages",
            "",
            f"{len(sets)} traced relationship(s), {n_papers} distinct papers.",
            f"Graph generated: {document.get('generated_at', 'unknown')}",
            "",
            "## Files",
            "",
            "| File | What it is |",
            "|---|---|",
            "| `lineages.md` | The routes in reading order, oldest first, with links |",
            "| `papers.csv` | One row per paper with every score component |",
            "| `papers.bib` | BibTeX for a reference manager |",
            "| `lineages.json` | The same structure, machine-readable |",
            "",
            "## What this is not",
            "",
            "- **Not the papers.** Most are paywalled and bulk-retrieving full texts",
            "  is not something this tool does. Every entry carries a resolvable DOI",
            "  or OpenAlex link instead.",
            "- **No abstracts.** A graph artifact stores identifiers, metrics and edges",
            "  only, so there is no body text to include.",
            "- **No author names.** The harvest fetches them; the graph does not yet",
            "  persist them, so the BibTeX entries have no `author` field.",
            "",
            "## How to read the numbers",
            "",
            "- **Influence is a rough guide.** It is a PageRank percentile computed",
            "  *within this harvest only* — not an absolute or authoritative measure.",
            "  The same paper scores differently under a different question.",
            "- A route is the path the citation record supports, not established",
            "  intellectual history.",
            leak_line,
            "- `rank_ci_low`/`rank_ci_high` give a bootstrap interval. A wide interval",
            "  means the ordering is unstable; treat it as such.",
            "- Publication years come from OpenAlex and reflect the indexed edition, so",
            "  a reprint can carry a much later year than the original work.",
            "",
        ]
    )


def lineage_bundle(
    document: Mapping[str, Any], sets: Sequence[LineageSet]
) -> dict[str, str]:
    """The bundle as ``path -> text``, all under one folder named for the query."""
    folder = slugify(str(document.get("query", "")))
    papers = _ordered_papers(document, sets)
    payload = {
        "query": document.get("query", ""),
        "generated_at": document.get("generated_at", ""),
        "boundary_leakage": document.get("boundary_leakage"),
        "lineages": [
            {"label": s.label, "kind": s.kind, "routes": [list(r) for r in s.routes]}
            for s in sets
        ],
        "papers": papers,
    }
    return {
        f"{folder}/README.md": readme(document, sets, len(papers)),
        f"{folder}/lineages.md": lineages_markdown(document, sets),
        f"{folder}/papers.csv": papers_csv(papers),
        f"{folder}/papers.bib": papers_bibtex(papers),
        f"{folder}/lineages.json": json.dumps(payload, indent=2) + "\n",
    }


def zip_bundle(files: Mapping[str, str]) -> bytes:
    """Deflate the bundle into a reproducible archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    return buffer.getvalue()
