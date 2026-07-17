# 0005 — Branch A trend engine without gensim

Status: Accepted · Date: 2026-07-11

## Context

Spec §6 names gensim's `LdaSeqModel` (Blei Dynamic Topic Model) as Branch A's
trend engine. On this project's pinned runtime (CPython 3.14, ADR-0001):

- pip resolves **no** modern gensim wheel for 3.14 — the only wheel it finds is
  `gensim 0.10.1` (2014, Python-2 era, no usable modern API);
- modern gensim (4.x) has C extensions and **fails to build** without Microsoft
  Visual C++ Build Tools, which are not present; even with them, gensim 4.x
  predates 3.14 and may not compile against its C API.

So `LdaSeqModel` as named is not feasible here without a heavy, uncertain
toolchain change or a Python downgrade that would disrupt the rest of the
(3.14-verified) codebase.

## Decision

Deliver the spec's **measurable outputs** — topic-probability drift, keyword
arrays, cluster data, and statistic-anchored trend descriptions (§§6, 10) — with
**scikit-learn**, which has 3.14 wheels (verified: scikit-learn 1.9.0, scipy
1.18.0).

Approach (`magnetor/trends.py`):
1. Fit one `LatentDirichletAllocation` topic space over the domain's abstracts.
2. Time-slice by `published` date; track each topic's mean prevalence per slice
   → the drift trajectories `LdaSeqModel` produces.
3. Keyword arrays per topic from the LDA components.
4. Anomaly detection = per-document term-rate spikes in the latest slice,
   normalised by ingestion volume (§6: growth ≠ novelty).
5. Statistic-anchored interpreter, templated and LLM-free (§6: no narrative-mood
   language).

## Consequences

- **Different algorithm, same deliverables.** This is not Blei's DTM — it's a
  per-slice LDA prevalence tracker. It reproduces the spec's *outputs* and
  corrections (volume-gated, drift-measuring, sentiment kept separate), not the
  exact probabilistic model. Documented so the deviation is explicit.
- **Faithful to the spec's intent** (§1: "dynamic topic modeling measures
  vocabulary and topic-probability drift, not affective tone").
- **New runtime deps:** scikit-learn + scipy (both 3.14 wheels).
- **Swappable:** if a 3.14-compatible gensim ever exists, the engine can be
  replaced behind `run_trend_analysis` without changing its callers or outputs.
- Trends need temporal spread to show drift; on a corpus of only recent papers
  there is one slice and the interpreter says so.
