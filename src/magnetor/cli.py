"""Command-line entry point (Spec: Scheduler / per-domain cadence).

Exposes ``magnetor acquire <domain>|all`` (intended to be driven by an OS
scheduler at each domain's cadence) and ``magnetor show <domain>`` (a read-only
view of stored records, for inspecting what acquisition produced). The cadence
gate lives in the pipeline, so scheduling more often than needed is harmless.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from magnetor.anchored import (
    DEFAULT_BACKWARD_HOPS,
    DEFAULT_FORWARD_FANOUT,
    NeighbourSource,
    OpenAlexNeighbours,
    run_anchored_harvest,
)
from magnetor.citations import SemanticScholarClient
from magnetor.config import (
    DomainConfig,
    all_domains,
    get_domain_config,
    global_store_path,
    resolve_domain,
)
from magnetor.deepdive import (
    DeepDiveResult,
    build_deep_dive,
    render_grounded_context,
)
from magnetor.deepdive import (
    Path as DeepDivePath,
)
from magnetor.embeddings.base import Embedder
from magnetor.embeddings.voyage import VoyageEmbedder
from magnetor.errors import MagnetorError
from magnetor.facets import classify, facet_counts
from magnetor.graph import build_graph_document, save_graph
from magnetor.graph_scoring import score_graph
from magnetor.harvest import (
    DEFAULT_EXPAND_PER_ROUND,
    HarvestResult,
    OpenAlexClient,
    WorksSource,
    run_harvest,
)
from magnetor.harvest import DEFAULT_LIMIT as HARVEST_DEFAULT_LIMIT
from magnetor.indexing import DEFAULT_BATCH_SIZE, EmbeddingResult, open_index, run_embedding
from magnetor.pipeline import DEFAULT_LIMIT, build_default_source, run_acquisition
from magnetor.relations import derive_relations
from magnetor.resources import DomainStore
from magnetor.robustness import DEFAULT_RESAMPLES, bootstrap_rank_cis
from magnetor.router import (
    DEFAULT_MARGIN,
    ROUTING_LOG_FILENAME,
    CrossDomainRouter,
    Hit,
    Routing,
    retrieve,
)
from magnetor.trends import (
    DEFAULT_NUM_TOPICS,
    DEFAULT_SLICE_DAYS,
    TrendResult,
    run_trend_analysis,
)
from magnetor.types import AcquisitionResult, Domain, Paper
from magnetor.vectors import VECTOR_FILENAME, stored_count

#: Characters of abstract to show per record unless ``--full`` is given.
_ABSTRACT_PREVIEW = 240


def _build_embedder() -> Embedder:
    """Construct the default embedder (indirection lets tests inject a fake)."""
    return VoyageEmbedder()


def _build_works_source() -> WorksSource:
    """Construct the default works source (indirection lets tests inject a fake)."""
    return OpenAlexClient()


def _build_neighbour_source() -> NeighbourSource:
    """Construct the default citation-neighbourhood source (fakeable in tests)."""
    return OpenAlexNeighbours()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "acquire":
        return _run_acquire(args)
    if args.command == "show":
        return _run_show(args)
    if args.command == "embed":
        return _run_embed(args)
    if args.command == "query":
        return _run_query(args)
    if args.command == "deepdive":
        return _run_deepdive(args)
    if args.command == "status":
        return _run_status(args)
    if args.command == "trends":
        return _run_trends(args)
    if args.command == "dashboard":
        return _run_dashboard(args)
    if args.command == "harvest":
        return _run_harvest(args)
    if args.command == "anchor":
        return _run_anchor(args)
    parser.print_help()
    return 2


def _run_harvest(args: argparse.Namespace) -> int:
    """Branch C (ADR-0006): harvest -> score -> robustness -> persist a graph."""
    try:
        result = run_harvest(
            _build_works_source(), args.query, limit=args.limit,
            expand_rounds=args.expand, expand_per_round=args.expand_top,
        )
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not result.papers:
        print(f"[harvest] no papers for {args.query!r}", file=sys.stderr)
        return 1
    return _persist_graph(result, args)


def _run_anchor(args: argparse.Namespace) -> int:
    """Branch C — build a graph outward from one paper rather than a question."""
    try:
        result = run_anchored_harvest(
            _build_neighbour_source(), args.paper, limit=args.limit,
            backward_hops=args.backward, forward_fanout=args.forward_fanout,
            expand_rounds=args.expand,
        )
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not result.papers:
        print(f"[anchor] no neighbourhood found for {args.paper!r}", file=sys.stderr)
        return 1
    print(f"[anchor] seed resolved to {result.query!r}")
    print(
        "  caveat: the seed ranks top by construction — the set was built from papers "
        "citing it, so its in-degree is an artefact of the gathering, not a finding. "
        "Read the ranking among the OTHER papers.",
        file=sys.stderr,
    )
    return _persist_graph(result, args)


def _persist_graph(result: HarvestResult, args: argparse.Namespace) -> int:
    """Score, measure robustness, derive relations, persist, and report.

    Shared by ``harvest`` and ``anchor``: the two differ only in how the working
    set is gathered, so everything after that is one code path.
    """
    scores = score_graph(result)
    robustness = bootstrap_rank_cis(result, resamples=args.resamples)
    # Derived relation layers (L4). Computed from reference lists already in
    # memory — no extra API calls — and kept out of the influence metric (D4).
    relations = derive_relations(result)
    # Facets are classified here, while abstracts are still in memory; only the
    # label and its evidence terms reach the document (section 3 / I4).
    facets = classify(result)
    document = build_graph_document(
        result, scores, robustness, top_n=args.top_n,
        relations=relations, facets=facets,
    )
    path = save_graph(document)

    leak = robustness.boundary_leakage
    if result.expansion_rounds:
        print(
            f"[harvest] {result.n_fetched} papers, {len(result.edges)} in-set edges; "
            f"leakage {result.seed_leakage:.0%} -> {leak:.0%} "
            f"after {result.expansion_rounds} expansion round(s)"
        )
    else:
        print(
            f"[harvest] {result.n_fetched} papers, {len(result.edges)} in-set edges; "
            f"boundary leakage {leak:.0%}"
        )
    drawn_coupled = len(document.get("biblio_coupled", []))
    drawn_cocited = len(document.get("co_cited", []))
    print(
        f"  relations: {drawn_coupled} bibliographic-coupling, {drawn_cocited} "
        "co-citation link(s) among the kept nodes (navigational only, not scored)"
    )
    spread = facet_counts(document["nodes"])
    if spread:
        summary = ", ".join(f"{facet} {count}" for facet, count in spread.items())
        print(f"  facets (multi-label, screening heuristic): {summary}")
    if leak >= 0.5:
        print(
            "  note: leakage still high — raise --expand / --expand-top for a more "
            "complete lineage. Shared external references are still usable as "
            "coupling links.",
            file=sys.stderr,
        )
    if result.self_referencing_ids:
        ids = ", ".join(result.self_referencing_ids)
        print(
            f"  flag: dropped {len(result.self_referencing_ids)} self-citation edge(s) "
            f"(upstream OpenAlex data error) on: {ids}",
            file=sys.stderr,
        )
    for node in document["nodes"][:5]:
        ci = f"CI {node['lo_rank']}-{node['hi_rank']}" if node["lo_rank"] else "-"
        title = node["title"][:52]
        print(f"  infl {node['influence']:.2f}  in={node['in_degree']:<3} {ci}  {title}")
    print(f"  saved -> {path}")
    return 0


def _dashboard_command(port: int | None) -> list[str]:
    """Build the ``streamlit run`` argv for the dashboard app (Spec 11)."""
    app = str(Path(__file__).with_name("dashboard.py"))
    cmd = [sys.executable, "-m", "streamlit", "run", app]
    if port is not None:
        cmd += ["--server.port", str(port)]
    return cmd


def _run_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard as a child process."""
    try:
        return subprocess.call(_dashboard_command(args.port))
    except FileNotFoundError:
        print(
            "error: streamlit is not installed. Install the dashboard extra: "
            "pip install -e '.[dashboard]'",
            file=sys.stderr,
        )
        return 2


def _run_trends(args: argparse.Namespace) -> int:
    try:
        domains = _selected_domains(args.domain)
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    exit_code = 0
    for domain in domains:
        config = get_domain_config(domain)
        store = DomainStore(domain, config.storage_dir)
        try:
            result = run_trend_analysis(
                config, store, num_topics=args.topics,
                slice_days=args.slice_days, force=args.force,
            )
        except MagnetorError as exc:
            print(f"[{domain.value}] error: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(_format_trends(result))
    return exit_code


def _format_trends(result: TrendResult) -> str:
    if not result.ran:
        return f"[{result.domain.value}] skipped: {result.reason}"
    head = f"[{result.domain.value}] docs={result.n_docs} slices={result.n_slices}"
    if not result.topics:
        return f"{head}  ({result.reason})"
    lines = [head]
    for topic in result.topics:
        kw = ", ".join(topic.keywords[:5])
        lines.append(f"  topic {topic.topic_id} (drift {topic.drift:+.3f}): {kw}")
    if result.interpretation:
        lines.append("  interpretation:")
        lines.extend(f"    - {line}" for line in result.interpretation)
    return "\n".join(lines)


def _run_deepdive(args: argparse.Namespace) -> int:
    forced: Domain | None = None
    if args.domain is not None:
        try:
            forced = resolve_domain(args.domain)
        except MagnetorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    embedder = _build_embedder()
    indices = {
        domain: open_index(get_domain_config(domain), embedder) for domain in all_domains()
    }
    router = CrossDomainRouter(indices, log_path=global_store_path(ROUTING_LOG_FILENAME))
    expander = SemanticScholarClient()
    try:
        result = build_deep_dive(
            embedder, router, args.query, expander,
            k=args.k, domain=forced, threshold=args.threshold,
        )
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.payload:
        print(render_grounded_context(result))
    else:
        print(_format_deep_dive(result))
    return 0


def _format_deep_dive(result: DeepDiveResult) -> str:
    if result.path is None:
        return "no results (has this domain been embedded? run: magnetor embed <domain>)"
    domain = result.domain.value if result.domain else "-"
    head = (
        f"[{domain}] {result.path.value}  "
        f"top={result.top_score:.3f} threshold={result.threshold:.3f}"
    )
    lines = [head]
    if result.path is DeepDivePath.ANCHOR_LOCK and result.anchor is not None:
        anchor = result.anchor
        lines.append(f"  anchor: {anchor.paper.title}")
        lines.append(
            f"  citations: {len(anchor.forward)} forward, {len(anchor.backward)} backward"
        )
    elif result.path is DeepDivePath.FIELD_MAP and result.field_map is not None:
        lines.append(f"  {len(result.field_map.positions)} competing positions:")
        for pos in result.field_map.positions:
            lines.append(f"    [{pos.rank}] {pos.score:.3f}  {pos.paper.title}")
    return "\n".join(lines)


def _run_acquire(args: argparse.Namespace) -> int:
    try:
        domains = _selected_domains(args.domain)
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    exit_code = 0
    for domain in domains:
        config = get_domain_config(domain)
        try:
            result = _acquire_one(config, force=args.force, limit=args.limit, dry_run=args.dry_run)
        except MagnetorError as exc:
            print(f"[{domain.value}] error: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(_format(result))
        if result.cold_start_empty:
            print(
                f"[{domain.value}] warning: first run fetched 0 records — "
                "check the source configuration",
                file=sys.stderr,
            )
    return exit_code


def _run_status(args: argparse.Namespace) -> int:
    try:
        domains = _selected_domains(args.domain)
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{'domain':12} {'records':>8} {'embedded':>9} {'last run':>20} {'watermark':>20}")
    for domain in domains:
        config = get_domain_config(domain)
        store = DomainStore(domain, config.storage_dir)
        embedded = stored_count(config.storage_dir / VECTOR_FILENAME)
        print(
            f"{domain.value:12} {store.record_count():>8} {embedded:>9} "
            f"{_fmt_ts(store.last_run()):>20} {_fmt_ts(store.watermark()):>20}"
        )
    return 0


def _fmt_ts(moment: dt.datetime | None) -> str:
    return moment.date().isoformat() if moment else "-"


def _run_embed(args: argparse.Namespace) -> int:
    try:
        domains = _selected_domains(args.domain)
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    embedder = _build_embedder()
    exit_code = 0
    for domain in domains:
        config = get_domain_config(domain)
        store = DomainStore(domain, config.storage_dir)
        try:
            index = open_index(config, embedder)
            result = run_embedding(store, index, embedder, batch_size=args.batch_size)
        except MagnetorError as exc:
            print(f"[{domain.value}] error: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(_format_embed(result))
    return exit_code


def _run_query(args: argparse.Namespace) -> int:
    forced: Domain | None = None
    if args.domain is not None:
        try:
            forced = resolve_domain(args.domain)
        except MagnetorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    embedder = _build_embedder()
    indices = {
        domain: open_index(get_domain_config(domain), embedder) for domain in all_domains()
    }
    router = CrossDomainRouter(
        indices, margin=args.margin, log_path=global_store_path(ROUTING_LOG_FILENAME)
    )
    try:
        routing, hits = retrieve(embedder, router, args.query, k=args.k, domain=forced)
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if routing is not None:
        print(_format_routing(routing))
    if not hits:
        print("no results (has this domain been embedded? run: magnetor embed <domain>)")
        return 0
    titles = _titles_for({hit.domain for hit in hits})
    for hit in hits:
        print(_format_hit(hit, titles.get((hit.domain, hit.external_id), "")))
    return 0


def _titles_for(domains: set[Domain]) -> dict[tuple[Domain, str], str]:
    titles: dict[tuple[Domain, str], str] = {}
    for domain in domains:
        store = DomainStore(domain, get_domain_config(domain).storage_dir)
        for paper in store.read_records():
            titles[(domain, paper.external_id)] = paper.title
    return titles


def _run_show(args: argparse.Namespace) -> int:
    try:
        domain = resolve_domain(args.domain)
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = get_domain_config(domain)
    store = DomainStore(domain, config.storage_dir)
    papers = store.read_records(limit=args.limit)
    if not papers:
        print(
            f"[{domain.value}] no records stored "
            f"(run: magnetor acquire {domain.value})"
        )
        return 0
    for paper in papers:
        print(_format_record(paper, full=args.full))
    return 0


def _acquire_one(
    config: DomainConfig,
    *,
    force: bool,
    limit: int,
    dry_run: bool,
) -> AcquisitionResult:
    store = DomainStore(config.domain, config.storage_dir)
    if dry_run:
        last = store.last_run()
        reason = "dry-run" if last is None else f"dry-run (last run {last.isoformat()})"
        return AcquisitionResult(
            domain=config.domain,
            fetched=0,
            stored=0,
            skipped_duplicates=0,
            ran=False,
            reason=reason,
        )
    source = build_default_source(config, store)
    return run_acquisition(config, source, store, force=force, limit=limit)


def _selected_domains(token: str) -> tuple[Domain, ...]:
    if token == "all":
        return all_domains()
    return (resolve_domain(token),)


def _format(result: AcquisitionResult) -> str:
    if not result.ran:
        return f"[{result.domain.value}] skipped: {result.reason}"
    return (
        f"[{result.domain.value}] fetched={result.fetched} "
        f"stored={result.stored} duplicates={result.skipped_duplicates}"
    )


def _format_routing(routing: Routing) -> str:
    scores = "  ".join(f"{s.domain.value}={s.score:.3f}" for s in routing.scored)
    selected = ", ".join(d.value for d in routing.selected) or "(none)"
    return f"router: {scores or '(no embedded domains)'}\nselected: {selected}\n"


def _format_hit(hit: Hit, title: str) -> str:
    return f"[{hit.domain.value}] {hit.score:.3f}  {title or hit.external_id}"


def _format_embed(result: EmbeddingResult) -> str:
    return (
        f"[{result.domain.value}] embedded={result.embedded} "
        f"(skipped existing={result.skipped_existing}, empty={result.skipped_empty}) "
        f"index_size={result.index_size}"
    )


def _format_record(paper: Paper, *, full: bool) -> str:
    published = paper.published.date().isoformat() if paper.published else "n/a"
    full_text = "yes" if paper.full_text_available else "no"
    authors = ", ".join(paper.authors) if paper.authors else "(unknown)"
    abstract = paper.abstract or "(none)"
    if not full and len(abstract) > _ABSTRACT_PREVIEW:
        abstract = abstract[:_ABSTRACT_PREVIEW].rstrip() + "..."
    return "\n".join(
        [
            f"[{paper.external_id}] {paper.title}",
            f"  published={published}  source={paper.source}  full_text={full_text}",
            f"  authors: {authors}",
            f"  abstract: {abstract}",
            "",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magnetor", description="Domain-Aware Research Platform")
    sub = parser.add_subparsers(dest="command")

    acquire = sub.add_parser("acquire", help="Run acquisition for a domain (or all).")
    acquire.add_argument(
        "domain",
        help="Domain token (qm, math, neuro, philosophy, anthro, history) or 'all'.",
    )
    acquire.add_argument(
        "--force",
        action="store_true",
        help="Bypass the cadence gate and run now.",
    )
    acquire.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max papers to fetch this run (default: {DEFAULT_LIMIT}).",
    )
    acquire.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would run without fetching or storing.",
    )

    embed = sub.add_parser(
        "embed", help="Embed a domain's stored abstracts into its vector index."
    )
    embed.add_argument(
        "domain",
        help="Domain token (qm, math, neuro, philosophy, anthro, history) or 'all'.",
    )
    embed.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Passages per embedding request (default: {DEFAULT_BATCH_SIZE}).",
    )

    status = sub.add_parser(
        "status", help="Show record counts, index sizes, and run state per domain."
    )
    status.add_argument(
        "domain",
        nargs="?",
        default="all",
        help="Domain token or 'all' (default: all).",
    )

    query = sub.add_parser(
        "query", help="Route a question and return the closest stored abstracts."
    )
    query.add_argument("query", help="The natural-language question to route and search.")
    query.add_argument(
        "--k", type=int, default=5, help="Results per selected domain (default: 5)."
    )
    query.add_argument(
        "--domain",
        default=None,
        help="Skip routing and search only this domain (qm, math, neuro, ...).",
    )
    query.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Route to a 2nd domain if within this cosine margin (default: {DEFAULT_MARGIN}).",
    )

    trends = sub.add_parser(
        "trends",
        help="Branch A: per-domain topic-trend tracking (Spec 6). Volume-gated.",
    )
    trends.add_argument("domain", help="Domain token or 'all'.")
    trends.add_argument("--force", action="store_true", help="Bypass the volume gate.")
    trends.add_argument(
        "--topics", type=int, default=DEFAULT_NUM_TOPICS, help="Number of topics (default: 5)."
    )
    trends.add_argument(
        "--slice-days", type=int, default=DEFAULT_SLICE_DAYS,
        help="Days per time slice for drift (default: 7).",
    )

    deepdive = sub.add_parser(
        "deepdive",
        help="Branch B: Anchor-Lock (single paper + citations) or Field-Map (Spec 7.2).",
    )
    deepdive.add_argument("query", help="The natural-language question to deep-dive.")
    deepdive.add_argument(
        "--k", type=int, default=5, help="Candidates retrieved before path selection (default: 5)."
    )
    deepdive.add_argument(
        "--domain", default=None, help="Skip routing and deep-dive only this domain."
    )
    deepdive.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the domain's Anchor-Lock threshold for this run (Spec 7.2).",
    )
    deepdive.add_argument(
        "--payload",
        action="store_true",
        help="Print the LLM-ready grounded context instead of the summary.",
    )

    dashboard = sub.add_parser(
        "dashboard",
        help="Launch the Streamlit dashboard (Spec 11): banner, viewport, frontier feed.",
    )
    dashboard.add_argument(
        "--port", type=int, default=None, help="Port for the Streamlit server."
    )

    harvest = sub.add_parser(
        "harvest",
        help="Branch C (ADR-0006): build an Evidence Graph for a query (offline batch).",
    )
    harvest.add_argument("query", help="The research question or keyword to harvest.")
    harvest.add_argument(
        "--limit", type=int, default=HARVEST_DEFAULT_LIMIT,
        help=f"Papers to harvest from OpenAlex (default: {HARVEST_DEFAULT_LIMIT}).",
    )
    harvest.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"Bootstrap resamples for rank CIs (default: {DEFAULT_RESAMPLES}).",
    )
    harvest.add_argument(
        "--top-n", type=int, default=60,
        help="Keep the top-N most influential nodes in the saved graph (default: 60).",
    )
    harvest.add_argument(
        "--expand", type=int, default=2,
        help="Snowball rounds pulling in missing foundational papers to cut "
             "boundary leakage (default: 2; 0 = seed only).",
    )
    harvest.add_argument(
        "--expand-top", type=int, default=DEFAULT_EXPAND_PER_ROUND,
        help=f"Foundational works pulled in per round (default: {DEFAULT_EXPAND_PER_ROUND}).",
    )

    anchor = sub.add_parser(
        "anchor",
        help="Branch C: build an Evidence Graph outward from ONE paper "
             "(DOI, OpenAlex id, or a link), to trace how it evolved.",
    )
    anchor.add_argument("paper", help="Seed paper: a DOI, an OpenAlex id (W...), or a URL.")
    anchor.add_argument(
        "--limit", type=int, default=HARVEST_DEFAULT_LIMIT,
        help=f"Papers in the neighbourhood (default: {HARVEST_DEFAULT_LIMIT}).",
    )
    anchor.add_argument(
        "--backward", type=int, default=DEFAULT_BACKWARD_HOPS,
        help=f"Hops toward antecedents (default: {DEFAULT_BACKWARD_HOPS}). Cheap - "
             "references arrive inline.",
    )
    anchor.add_argument(
        "--forward-fanout", type=int, default=DEFAULT_FORWARD_FANOUT,
        help="First-hop citing papers that are themselves expanded forward "
             f"(default: {DEFAULT_FORWARD_FANOUT} = seed only). Each one costs a query.",
    )
    anchor.add_argument(
        "--expand", type=int, default=0,
        help="Snowball rounds (default: 0). An anchored set is already closed "
             "around its seed, so expansion mostly adds unrelated foundations.",
    )
    anchor.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"Bootstrap resamples for rank CIs (default: {DEFAULT_RESAMPLES}).",
    )
    anchor.add_argument(
        "--top-n", type=int, default=60,
        help="Keep the top-N most influential nodes in the saved graph (default: 60).",
    )

    show = sub.add_parser("show", help="Print stored records for a domain (read-only).")
    show.add_argument(
        "domain",
        help="Domain token (qm, math, neuro, philosophy, anthro, history).",
    )
    show.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max records to print, newest first (default: 10).",
    )
    show.add_argument(
        "--full",
        action="store_true",
        help="Print full abstracts instead of a truncated preview.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
