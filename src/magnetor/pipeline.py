"""Acquisition orchestration (Spec Section 4, Phase 1).

Ties one domain's config, source, and store together into a single run:

    cadence gate -> fetch since last run -> store (within-domain dedup) -> record

The source and store are injected, so this module never touches the network or
the real filesystem directly and is fully unit-testable with fakes.
"""

from __future__ import annotations

import datetime as dt

import httpx

from magnetor.config import DomainConfig
from magnetor.interfaces import DomainSource, SupportsPostRun
from magnetor.resources import DomainStore
from magnetor.sources.arxiv import ArxivSource
from magnetor.sources.manual_batch import INBOX_DIRNAME, ManualBatchSource
from magnetor.sources.pubmed import PubMedCentralSource
from magnetor.types import AcquisitionResult, Domain

DEFAULT_LIMIT = 100


def run_acquisition(
    config: DomainConfig,
    source: DomainSource,
    store: DomainStore,
    *,
    now: dt.datetime | None = None,
    force: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> AcquisitionResult:
    """Run one acquisition cycle for a domain.

    Args:
        config: Static config for the domain.
        source: Bound source to fetch from (injected for testability).
        store: Bound store to persist into (injected for testability).
        now: Wall-clock override (defaults to current UTC time).
        force: Bypass the cadence gate for a manual/one-off run.
        limit: Maximum papers to fetch this run.

    Returns:
        An :class:`AcquisitionResult` summarising what happened. A cadence skip
        is a normal, non-error outcome with ``ran=False``.
    """
    moment = now or dt.datetime.now(tz=dt.UTC)
    last_attempt = store.last_run()

    if not force and last_attempt is not None and (moment - last_attempt) < config.cadence:
        return AcquisitionResult(
            domain=config.domain,
            fetched=0,
            stored=0,
            skipped_duplicates=0,
            ran=False,
            reason=f"cadence gate: {config.cadence} not elapsed since {last_attempt.isoformat()}",
        )

    fetched = 0
    stored = 0
    duplicates = 0
    newest_seen: dt.datetime | None = None
    # `since` is the ingestion watermark, NOT the last attempt: a prior empty or
    # failed run must not skip a window it never actually covered.
    watermark_before = store.watermark()
    known = store.known_keys()
    for paper in source.fetch(since=watermark_before, limit=limit):
        fetched += 1
        if paper.published is not None and (newest_seen is None or paper.published > newest_seen):
            newest_seen = paper.published
        if paper.dedup_key() in known:
            duplicates += 1
            continue
        if store.store(paper):
            stored += 1
            known.add(paper.dedup_key())
        else:
            duplicates += 1

    # Cap the watermark at the run time: some sources emit future publication
    # dates (e.g. PubMed ahead-of-print issue dates), which would otherwise push
    # `since` past today and skip every paper until that future date arrives.
    if newest_seen is not None and newest_seen > moment:
        newest_seen = moment

    store.record_run(moment, stored=stored, watermark=newest_seen)
    if isinstance(source, SupportsPostRun):
        source.on_run_complete()
    return AcquisitionResult(
        domain=config.domain,
        fetched=fetched,
        stored=stored,
        skipped_duplicates=duplicates,
        ran=True,
        cold_start_empty=watermark_before is None and fetched == 0,
    )


def build_default_source(
    config: DomainConfig,
    store: DomainStore,
    *,
    client: httpx.Client | None = None,
) -> DomainSource:
    """Construct the real source for a domain, wired per Spec Sections 3-4.

    Raises:
        ValueError: The domain has no default source wiring (unreachable while
            the registry and this factory stay in sync).
    """
    domain = config.domain
    if domain in (Domain.QUANTUM_MECHANICS, Domain.MATHEMATICS):
        return ArxivSource(domain, client=client)
    if domain is Domain.NEUROSCIENCE:
        return PubMedCentralSource(client=client)
    if domain in (Domain.PHILOSOPHY, Domain.ANTHROPOLOGY, Domain.HISTORY):
        return ManualBatchSource(domain, store.root / INBOX_DIRNAME)
    raise ValueError(f"No default source wiring for domain {domain!r}")
