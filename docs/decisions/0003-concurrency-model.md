# 0003 — Concurrency model for Phase 1

Status: Accepted · Date: 2026-07-04

## Context

The architecture spec treats concurrency as a deliberate choice per workflow,
not a default. Phase 1 is an acquisition workload: mostly blocking network I/O
against polite, rate-limited public APIs (arXiv, NCBI E-utilities), plus local
file writes.

## Decision

**Phase 1 is synchronous and single-process.** No threads, async, or
multiprocessing are introduced.

Rationale:

- The workload is bounded and latency-tolerant. Acquisition runs on a schedule
  (cron / Task Scheduler) at each domain's cadence; wall-clock speed is not a
  goal.
- Upstream sources demand politeness (NCBI rate limits, arXiv usage terms).
  Serial requests are the safe default; parallelism would risk throttling or
  bans for no real benefit at this stage.
- The cadence gate lives in the pipeline, so scheduling each domain
  independently already gives coarse-grained parallelism at the OS level if
  wanted — run six scheduled jobs, one per domain.

## Boundaries kept open for later

- The HTTP boundary (`sources/_http.py`) accepts an injected `httpx.Client`.
  httpx has an async client with the same surface, so a future high-throughput
  path can move to `async`/`await` without reshaping the sources.
- Domains are fully isolated, so a later phase could fan out acquisition across
  processes (one per domain) with zero shared state — consistent with the
  spec's "use processes/executors for true parallelism" guidance.

## Consequences

- Simple, debuggable control flow now; the expensive concurrency decision is
  deferred until a measured need appears (e.g. full-text bulk download volume).
