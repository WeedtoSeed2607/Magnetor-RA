# Magnetor

**Domain-Aware Research Platform** — Phase 1: the Acquisition layer of the
Revised Architecture Specification v2.1.

> Core principle: *Isolated Storage, Federated Retrieval* — "silo the data, not
> the answer." No document body or embedding is ever copied across domain
> directories.

## What Phase 1 does

Pulls papers per domain into physically isolated per-domain storage, honouring
each source's real cadence and redistribution terms (Spec Sections 3–4).

| Group | Domains | Source | Cadence | Mode |
|-------|---------|--------|---------|------|
| Fast  | Quantum Mechanics, Mathematics | arXiv | daily | automated bulk API |
| Fast  | Neuroscience | PubMed Central | daily | automated bulk API |
| Slow  | Philosophy, Anthropology, History | PhilPapers OA / AnthroSource / JSTOR DfR + OpenAlex | 7–30 days | operator drop-folder batch |

The slow group has no real-time bulk API, so acquisition there is operator-driven:
a human runs the mediated request or aggregator query and drops metadata records
as JSON into the domain's `_inbox/`. Full text is stored **only** when a record
carries an explicit `license`; otherwise it is refused (`RedistributionError`).

### Explicitly out of Phase 1 scope

Cross-domain router, trend engine (Branch A), Critic Agent, pedagogical layer,
vector index, and the global `dedup_index.json` are later phases. Within-domain
idempotency (a paper isn't re-stored) is handled; cross-*domain* dedup is not.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows
# source .venv/bin/activate on POSIX
```

Requires Python **3.14+**.

## Usage

```bash
magnetor acquire qm                 # one domain
magnetor acquire all                # every domain, cadence-gated
magnetor acquire neuro --force      # bypass the cadence gate
magnetor acquire history --dry-run  # report without fetching
```

Storage root defaults to a per-user, always-writable location
(`%LOCALAPPDATA%\Magnetor\data` on Windows, `~/.magnetor/data` on POSIX);
override with `MAGNETOR_DATA_ROOT`. The spec's `/data/<domain>` is illustrative —
on Windows it would resolve to `C:\data` and need admin rights, so it is not the
default.

On Windows you can run without activating the venv (avoids execution-policy and
PATH issues) via the bundled launcher:

```cmd
mag.cmd acquire qm --force --limit 3
```
NCBI etiquette: set `MAGNETOR_NCBI_EMAIL` (and optionally
`MAGNETOR_NCBI_API_KEY`) so PubMed requests are identifiable.

Intended to run under an OS scheduler (cron / Task Scheduler) at each domain's
cadence; the cadence gate makes over-scheduling a harmless no-op.

## Reliability & politeness

Live sources go through one HTTP boundary (`sources/_http.py`) that:
- **spaces requests** under each API's rate limit (arXiv ~3s between requests;
  NCBI 3 req/s without a key, 10 with one — set `MAGNETOR_NCBI_API_KEY`);
- **retries** transient failures (HTTP 429/5xx) with backoff, honouring
  `Retry-After`, then surfaces a clean `SourceUnavailableError` on exhaustion.

Slow-group drops are moved to `_inbox/_archive/` after a successful run, so the
inbox never re-parses the same files. PubMed records are enriched with abstracts
via `efetch` (esummary omits them).

## Development

```bash
.venv/Scripts/python -m pytest      # tests (network fully mocked)
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m mypy
```

See `docs/decisions/` for the version/style, data-flow, and concurrency ADRs.
