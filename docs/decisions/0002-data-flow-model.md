# 0002 — Data-flow model for acquisition

Status: Accepted · Date: 2026-07-04

## Context

Spec Sections 3–4 require per-domain isolated storage and per-source cadence,
with the load-bearing invariant that no document body or embedding is ever
copied across domain directories. Phase 1 must realise the acquisition slice of
this without prematurely building later-phase machinery.

## Decision

- **Records stream, they do not accumulate globally.** Each source yields
  `Paper` objects lazily (`Iterable[Paper]`); the pipeline consumes them one at
  a time and persists per domain. No cross-domain in-memory collection exists.
- **`Paper` is a frozen, source-agnostic dataclass.** Sources normalise their
  own formats (arXiv Atom, PMC E-utilities JSON, manual JSON drops) into this
  one shape at the boundary, so the pipeline and store never branch on origin.
- **Storage is filesystem-per-domain.** `DomainStore` is bound to one domain
  directory and guards every write against escaping that root, restating the
  isolation rule mechanically. Metadata is written as one JSON file per record;
  run state lives in `_state.json`.
- **Idempotency is within-domain.** A `dedup_key` (DOI when present, else
  `source:external_id`) prevents re-storing the same paper on repeated runs.
  Cross-*domain* dedup (Spec Section 10 `dedup_index.json`) is deferred — see
  "Out of scope."
- **Incremental fetch.** The store records `last_run`; the next run passes it as
  `since`, so sources only pull new material. Cold start uses a small source-
  chosen lookback window.

## Out of scope (later phases)

Vector index, embeddings, the global `dedup_index.json`, trend/DTM outputs, and
sentiment stores are intentionally absent. Phase 1 stops at acquired metadata +
run state.

## Consequences

- Full text is a *pointer* in Phase 1 (`pdf_url`, `full_text_available`), not a
  stored blob; bulk full-text retrieval is a later pass.
- Because records stream and storage is per-domain, adding a domain is a config
  entry plus a source module — no changes to the pipeline.
