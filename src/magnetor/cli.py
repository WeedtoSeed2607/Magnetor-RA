"""Command-line entry point (Spec: Scheduler / per-domain cadence).

Exposes ``magnetor acquire <domain>|all`` (intended to be driven by an OS
scheduler at each domain's cadence) and ``magnetor show <domain>`` (a read-only
view of stored records, for inspecting what acquisition produced). The cadence
gate lives in the pipeline, so scheduling more often than needed is harmless.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
from magnetor.gaps import WorkFetcher, enrich, foundational_gaps, gap_rows
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
from magnetor.naming import (
    VERDICT_PROMPT,
    VERDICTS,
    NamingSet,
    agreement,
    bundle,
    freeze,
    import_lines,
    list_sets,
    load_set,
    names,
    to_judge,
    to_name,
    visible,
    write_judgement,
    write_name,
)
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
from magnetor.themes import DEFAULT_THEMES, discover_themes
from magnetor.trends import (
    DEFAULT_NUM_TOPICS,
    DEFAULT_SLICE_DAYS,
    TrendResult,
    run_trend_analysis,
)
from magnetor.types import AcquisitionResult, Domain, Paper
from magnetor.vectors import VECTOR_FILENAME, load_embeddings, stored_count

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
    if args.command == "name":
        return _run_name(args)
    parser.print_help()
    return 2


def _run_harvest(args: argparse.Namespace) -> int:
    """Branch C (ADR-0006): harvest -> score -> robustness -> persist a graph."""
    source = _build_works_source()
    try:
        result = run_harvest(
            source, args.query, limit=args.limit,
            expand_rounds=args.expand, expand_per_round=args.expand_top,
        )
    except MagnetorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not result.papers:
        print(f"[harvest] no papers for {args.query!r}", file=sys.stderr)
        return 1
    return _persist_graph(result, args, source)


def _resolve_set(set_id: str | None) -> NamingSet | None:
    """The set asked for, or the only one there is. Never a guess between several."""
    if set_id:
        return load_set(set_id)
    available = list_sets()
    if len(available) == 1:
        return available[0]
    if not available:
        print("No frozen sets yet. Run `magnetor name freeze --domain <token>` first.")
        return None
    print("Several frozen sets exist. Name one with --set:")
    for candidate in available:
        print(f"  {candidate.set_id}  {candidate.domain}  "
              f"{len(candidate.clusters)} groups  {candidate.created_at[:10]}")
    return None


def _cluster_titles(
    domain_token: str, paper_ids: Sequence[str], limit: int
) -> list[str]:
    """Titles for a frozen cluster — the only thing a reader is shown while naming.

    ``limit`` of 0 means every paper. Truncating is offered but not the default:
    a reader shown ten of thirty-five names those ten, and two readers would then
    be agreeing about an excerpt rather than about the group.
    """
    domain = resolve_domain(domain_token)
    config = get_domain_config(domain)
    store = DomainStore(domain, config.storage_dir)
    titles = {p.external_id: p.title for p in store.read_records()}
    wanted = paper_ids if limit <= 0 else paper_ids[:limit]
    return [titles.get(pid, f"({pid} — title not stored)") for pid in wanted]


def _ask(prompt: str) -> str:
    """Read one line, treating EOF as an empty answer so a pipe ends cleanly.

    Uses :func:`visible` rather than ``strip``: a pasted or piped line can carry a
    byte-order mark, which is not whitespace and would read as a real answer.
    """
    try:
        return visible(input(prompt))
    except EOFError:
        return ""


def _run_name(args: argparse.Namespace) -> int:
    if args.step == "freeze":
        return _run_name_freeze(args)
    naming_set = _resolve_set(args.set_id)
    if naming_set is None:
        return 1
    if args.step == "write":
        return _run_name_write(args, naming_set)
    if args.step == "judge":
        return _run_name_judge(args, naming_set)
    if args.step == "bundle":
        return _run_name_bundle(args, naming_set)
    if args.step == "import":
        return _run_name_import(args, naming_set)
    return _run_name_report(naming_set)


def _run_name_freeze(args: argparse.Namespace) -> int:
    if not args.domain:
        print("freeze needs --domain.")
        return 2
    domain = resolve_domain(args.domain)
    config = get_domain_config(domain)
    ids, vectors = load_embeddings(config.storage_dir / VECTOR_FILENAME)
    store = DomainStore(domain, config.storage_dir)
    papers = {p.external_id: p for p in store.read_records()}
    report = discover_themes(ids, vectors, papers, n_themes=args.clusters)
    if not report.usable:
        print(f"Nothing to freeze: {report.reason}")
        return 1
    frozen = freeze(
        domain.value,
        [(theme.members, theme.label) for theme in report.themes],
    )
    print(f"Frozen {len(frozen.clusters)} group(s) from {domain.value} "
          f"as set {frozen.set_id}.")
    print("The membership is now fixed, so every reader names the same groups.")
    print(f"\nNext: `magnetor name write --set {frozen.set_id} --as <your-name>`")
    return 0


def _run_name_write(args: argparse.Namespace, naming_set: NamingSet) -> int:
    pending = to_name(naming_set, coder=args.coder)
    if not pending:
        print("You have named every group in this set.")
        print(f"Next: `magnetor name judge --set {naming_set.set_id} "
              f"--as {args.coder or '<your-name>'}`")
        return 0
    print(f"{len(pending)} group(s) left to name. One short line each — what are "
          f"these papers about?")
    print("The computed keywords are deliberately not shown: they would be easier "
          "to paraphrase than the papers are to read.\n")
    for cluster in pending:
        print("=" * 72)
        for title in _cluster_titles(naming_set.domain, cluster.paper_ids, args.titles):
            print(f"  - {title}")
        extra = len(cluster.paper_ids) - args.titles
        if extra > 0:
            print(f"  ... and {extra} more in this group")
        answer = _ask("\nName for this group (blank to stop): ")
        if not answer:
            print("Stopped. Your finished groups are saved.")
            return 0
        write_name(naming_set, cluster.cluster_id, answer, coder=args.coder)
    print("\nAll groups named.")
    return 0


def _run_name_judge(args: argparse.Namespace, naming_set: NamingSet) -> int:
    pending = to_judge(naming_set, judge=args.coder)
    if not pending:
        print("Nothing left to rule on. Run `magnetor name report` for the figure.")
        return 0
    keys = {v[0]: v for v in VERDICTS}
    print(f"{len(pending)} name(s) to rule on. You will see your own among them, "
          "unmarked — that is deliberate.\n")
    for candidate in pending:
        cluster = naming_set.cluster(candidate.cluster_id)
        if cluster is None:
            continue
        print("=" * 72)
        for title in _cluster_titles(naming_set.domain, cluster.paper_ids, args.titles):
            print(f"  - {title}")
        print(f'\n  Proposed name:  "{candidate.text}"')
        for verdict in VERDICTS:
            print(f"    [{verdict[0]}] {VERDICT_PROMPT[verdict]}")
        answer = _ask("\nYour ruling (a/p/r, blank to stop): ").lower()
        if not answer:
            print("Stopped. Your finished rulings are saved.")
            return 0
        if answer[0] not in keys:
            print("Not one of a/p/r — stopping rather than recording a guess.")
            return 1
        write_judgement(naming_set, candidate, keys[answer[0]], judge=args.coder)
    print("\nAll names ruled on.")
    return 0


def _run_name_bundle(args: argparse.Namespace, naming_set: NamingSet) -> int:
    """Write the self-contained file the public naming app reads."""
    domain = resolve_domain(naming_set.domain)
    config = get_domain_config(domain)
    store = DomainStore(domain, config.storage_dir)
    titles = {p.external_id: p.title for p in store.read_records()}
    target = Path(args.out) if args.out else Path("naming_bundle.json")
    seed = names(naming_set.set_id) if args.with_names else ()
    payload = bundle(naming_set, titles, seed=seed)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    clusters = payload["clusters"]
    papers = payload["titles"]
    print(f"Wrote {target} — {len(clusters) if isinstance(clusters, list) else 0} "
          f"group(s), {len(papers) if isinstance(papers, dict) else 0} titles.")
    print("It carries cluster membership and titles only: no abstracts, no keys, "
          "no corpus, and not the computed keywords.")
    if args.with_names:
        seeded = payload["seed"]
        print(f"Carrying {len(seeded) if isinstance(seeded, list) else 0} name(s) "
              "so remote readers can judge. Only do this once everyone has "
              "finished writing — a bundle holding names is one that shows them.")
    print("\nDeploy `naming_app.py` with this file beside it, then share the link.")
    return 0


def _run_name_import(args: argparse.Namespace, naming_set: NamingSet) -> int:
    """Merge a remote reader's returned answers."""
    source = Path(args.file)
    if not source.exists():
        print(f"No such file: {source}")
        return 1
    added, skipped = import_lines(naming_set.set_id, source.read_text(encoding="utf-8"))
    print(f"Merged {added} record(s); skipped {skipped} already present or unreadable.")
    if added:
        print(f"Run `magnetor name report --set {naming_set.set_id}` for the figure.")
    return 0


def _run_name_report(naming_set: NamingSet) -> int:
    result = agreement(naming_set.set_id)
    print(f"Set {naming_set.set_id} · {naming_set.domain} · "
          f"{len(naming_set.clusters)} groups")
    print(result.summary())
    if result.single_judged:
        print(f"{result.single_judged} name(s) have only one ruling and cannot "
              "contribute to the figure.")
    if result.verdict_spread:
        spread = ", ".join(f"{k} {v}" for k, v in result.verdict_spread.items())
        print(f"Rulings so far: {spread}")
    if result.alpha is not None:
        print(
            "\nAbove 0.80 the readers agree firmly; 0.67 to 0.80 is usable; below "
            "that the disagreement is about the question, not the readers — and "
            "that is a finding, not a failure."
        )
    return 0


def _run_anchor(args: argparse.Namespace) -> int:
    """Branch C — build a graph outward from one paper rather than a question."""
    source = _build_neighbour_source()
    try:
        result = run_anchored_harvest(
            source, args.paper, limit=args.limit,
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
    return _persist_graph(result, args, source)


def _persist_graph(
    result: HarvestResult, args: argparse.Namespace, source: WorkFetcher | None = None
) -> int:
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
    # Foundational gaps: outside works the set repeatedly leans on. Identifiers
    # come free from the reference lists; titles need one batch fetch, since by
    # definition these works were never retrieved.
    found_gaps = foundational_gaps(result)
    if found_gaps and source is not None:
        try:
            found_gaps = enrich(
                found_gaps, source.fetch_by_ids([g.openalex_id for g in found_gaps])
            )
        except MagnetorError as exc:  # a gap list without titles still beats none
            print(f"  note: gap titles unavailable ({exc})", file=sys.stderr)
    document["foundational_gaps"] = gap_rows(found_gaps)
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
    rows = document.get("foundational_gaps") or []
    if rows:
        print(
            f"  gaps: {len(rows)} outside work(s) cited by 2+ harvested papers but "
            f"absent from the map; the strongest by {rows[0][1]} paper(s)"
        )
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

    name = sub.add_parser(
        "name",
        help="Stage 1: measure whether two readers agree on what a group is about.",
    )
    name.add_argument(
        "step",
        choices=("freeze", "write", "judge", "report", "bundle", "import"),
        help=(
            "freeze: snapshot clusters once. write: name them. judge: rule on every "
            "name, unattributed. report: the agreement figure. bundle: write the "
            "file the public app reads. import: merge a remote reader's answers."
        ),
    )
    name.add_argument(
        "--out", default=None,
        help="Where to write the bundle (bundle only; default naming_bundle.json).",
    )
    name.add_argument(
        "--file", default=None,
        help="The .jsonl a remote reader sent back (import only).",
    )
    name.add_argument(
        "--with-names", action="store_true",
        help="Carry every name into the bundle so remote readers can judge "
             "(bundle only). Use once naming is finished, never before.",
    )
    name.add_argument(
        "--domain", default=None, help="Domain to freeze clusters from (freeze only)."
    )
    name.add_argument(
        "--set", dest="set_id", default=None,
        help="Which frozen set to work on. Defaults to the only one, if there is one.",
    )
    name.add_argument(
        "--clusters", type=int, default=DEFAULT_THEMES,
        help=f"Groups to freeze (default: {DEFAULT_THEMES}).",
    )
    name.add_argument(
        "--titles", type=int, default=0,
        help="Cap the titles shown per group. Default 0 = show every paper, "
             "because naming a group from an excerpt names the excerpt.",
    )
    name.add_argument(
        "--as", dest="coder", default=None,
        help="Who is working. Overrides MAGNETOR_CODER, so two people can share a machine.",
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
