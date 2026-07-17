# 0004 — Anchor-Lock threshold calibration (Spec §7.2)

Status: Accepted · Date: 2026-07-11

## Context

Spec §7.2 gates Branch B's Anchor-Lock path on `score >= domain-specific
calibrated threshold`, and explicitly:

- **removed the old flat 95%** ("unreachable in practice", §7.1),
- requires the value be **per-domain**, "not a flat global value",
- says it must be **derived from a labeled query-paper validation set**, "logged
  and reviewed periodically".

We do **not** have a labeled validation set yet, so a label-derived threshold is
not possible. But we do have empirical score distributions.

## Measurement

Batch on 2026-07-11: 18 fixed queries (6 per domain), each forced to its domain,
top-1 cosine recorded. Embedder: voyage-4-lite (asymmetric query/passage).

| | overall | qm | math | neuro |
|---|---|---|---|---|
| median | 0.489 | 0.546 | 0.466 | 0.471 |
| Q1 / Q3 | 0.455 / 0.547 | — | — | — |
| max | 0.596 | 0.596 | 0.577 | 0.549 |

The distribution is tight (std 0.076, IQR 0.093) and ceilings near ~0.60 —
confirming §7.1's point that question↔abstract cosines never approach 0.95.

## Decision

Set each **data-bearing domain's** Anchor-Lock threshold to its **observed
top-1 median** (rounded): `qm=0.55`, `math=0.47`, `neuro=0.47`. Humanities
domains (no corpus yet) keep the shared default `0.48` (≈ overall median).

At the median, Anchor-Lock fires for roughly the above-median half of on-topic
queries — a meaningful "confident single paper" bar that is neither unreachable
(0.95) nor trivial. `--threshold` overrides per run.

## Consequences and caveats

- **This is magnitude-informed, not label-calibrated.** It reflects how *similar*
  top hits score, not whether they are *correct* (we have no relevance labels).
  It is a principled placeholder, not the calibrated value §7.2 ultimately wants.
- **Recalibration path:** once a labeled query→expected-paper set exists per
  domain, replace these medians with values chosen against precision/recall on
  that set, and record the new basis here.
- Values live in `config.py` `DomainConfig.anchor_threshold` and are surfaced in
  every deep-dive result (`threshold=`), satisfying the "logged and reviewed"
  intent.
