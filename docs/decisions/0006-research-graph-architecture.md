# ADR-0006: Research Influence Graph — architecture

**Status:** Proposed (not implemented)
**Date:** 2026-07-19
**Supersedes:** the "point-7 Outlook board" concept (etymology / history / philosophical
roots), withdrawn because generated humanistic claims are hallucination-prone and
unverifiable. Etymology and philosophy return later as *rhetorical questions* framed
by measured results (§L5.4) — never as generated fact.

## 1. Context

The prior direction asked an LLM to *generate* conceptual history. This one inverts the
epistemics: build on **observable bibliometric data**, and use a model only for
*classification and description of grounded sets*, never for factual invention.

The user's specification, restated:

1. Query papers by a specific keyword/enquiry, over a corpus large enough to matter.
2. Order/weight/organise them by **classes of influence** (citations, author indices,
   journal tiers by international/national/regional reach, community discussion). The
   weighting hierarchy is decided *after* analysis, not assumed.
3. Build a **node network** linking papers through those classes; node size = influence
   with respect to the query.
4. **Even if the metrics are imperfect, a traceable pathway of papers must remain.**
5. Below the map, a summary + **pie chart of the theories** addressing the query,
   produced by a transformer/LLM.

## 2. Invariants (these govern every later decision)

- **I1 — Traceability over accuracy.** Every node, edge, score component and chart slice
  must resolve to concrete source records. A wrong weight must still leave a correct,
  inspectable graph. This is the user's point 4 promoted to an architectural law.
- **I2 — No synthetic edges.** Edges represent *real citation relations only*. If
  similarity edges are ever added they must be a structurally and visually distinct
  layer, never mixed into the citation backbone.
- **I3 — Grounded classification, not generation.** The model assigns papers to labels
  and describes the resulting sets. It does not assert facts about the field.
- **I4 — Isolation preserved.** Global artifacts (metrics cache, graph) store
  *identifiers, numbers and edges only*. Document bodies and embeddings never leave
  their domain directory, consistent with the core principle and with the existing
  global `routing_log.jsonl` / `topic_trend_log.json` precedent.
- **I5 — Offline compute, dashboard reads.** Graph construction is a batch job writing
  an artifact; the dashboard renders that artifact. Mirrors the existing trends design
  and is mandatory for hosted deployment.

## 3. Layered architecture

### L0 — Query and corpus selection
- Input: a **specific question sentence** (specificity is load-bearing for L5).
- Reuse: `VoyageEmbedder.embed_query` → `CrossDomainRouter` → per-domain search.
- Change: retrieval widens from deep-dive's `k=5` to a **candidate set** of N≈100–300.
  This is a distinct operation from `build_deep_dive`, not a parameter tweak.
- Output: `CandidateSet` — paper ids, domain, query-relevance (cosine).

### L1 — Bibliometric enrichment
- Boundary: `MetricsProvider` Protocol (mirrors `Embedder`, `CitationExpander`).
- Implementations: Semantic Scholar (citations, references, `influentialCitationCount`,
  venue, year, fields, open-access) and/or OpenAlex (`cited_by_count`, concepts,
  host venue, institutions). Crossref optional for DOI metadata.
- Persistence: **global** `metrics/` cache keyed by external id — metadata only (I4).
- Constraints: aggressive caching, batch endpoints where available, reuse the existing
  `Throttle`, and **resumable** builds. Rate limits, not CPU, are the bottleneck.

### L2 — Influence classes and weighting
- Boundary: `InfluenceClass` Protocol — `name`, `score(paper, metrics) -> float` in
  `[0,1]`, plus the provenance of what it consumed (I1).
- Initial classes: citation count, influential citations, venue tier, author index,
  community discussion, query relevance.
- **Weights are declared data, not code** — a versioned weight vector, swappable, with
  a sensitivity analysis (does ranking survive weight perturbation?). This honours
  "hierarchy decided after analysis."
- **Composite score must combine global influence with query relevance.** A
  multiplicative form is proposed:
  `size ∝ relevance^α · influence^β`
  so a heavily-cited but off-topic paper cannot dominate a keyword map. Purely additive
  scoring would reduce the map to "the field's most-cited papers," ignoring the query.

### L3 — Graph construction
- **Nodes:** candidate papers, optionally plus 1-hop citation neighbours.
- **Edges:** real citation relations (A cites B), directed, each carrying source and
  retrieval timestamp (I1, I2). This *is* the traceable pathway of point 4.
- **Node size:** L2 composite. **Node grouping/colour:** domain or theory label (L5).
- Output: a persisted `graph.json` (nodes, edges, per-class score breakdown,
  provenance, weight-vector version, build timestamp).

### L4 — Visualisation
- Interactive node-link map in the dashboard.
- Hover reveals the **per-class score breakdown** — the user must see *why* a node is
  large (auditability, I1). Click opens the paper via the existing link helpers.
- Rendering: a graph library with Python 3.14 wheels; `networkx` (pure-Python) for
  algorithms, plotting via Plotly or a custom component. **Wheel availability must be
  verified before commitment** — the gensim/ADR-0005 lesson.

### L5 — Theory classification and summary
- L5.1 Derive candidate theory labels by clustering candidate abstracts.
- L5.2 **Classify** each paper into a label (transformer/LLM). Classification, not
  generation (I3).
- L5.3 Pie/□ chart = share of papers per label. **Every slice click-throughs to its
  constituent papers** — traceability extended to the chart (I1).
- L5.4 Summary text describes each grounded set, constrained by the
  `render_grounded_context` `[GROUNDED]` discipline already built. Etymological /
  philosophical material appears **only as rhetorical questions** raised by the observed
  structure ("what explains the divergence between cluster A and B?"), never as claims.

### L6 — Evaluation (built now, not later)
- A `judgments/` store: researchers mark papers relevant/irrelevant and rate whether the
  top-ranked nodes are genuinely seminal.
- Metrics: precision@k for retrieval; Spearman correlation between composite score and
  researcher-judged importance.
- This closes the project's largest existing gap — 135 tests verify mechanics, nothing
  yet measures answer quality.

## 4. Methodological hazards (must be handled, not assumed away)

- **H1 — Citation counts are age-biased.** Raw counts make "most influential" collapse
  into "oldest." Default to **citations per year** and/or field-percentile
  normalisation; retain raw counts for display.
- **H2 — Field bias.** Biomedical citation rates dwarf mathematics. Cross-domain node
  sizes are not comparable without within-field normalisation.
- **H3 — Journal tiers are not canonically available.** Impact Factor (JCR) is
  proprietary. Scimago SJR is free and carries quartile + country, but mapping quartiles
  to "international / national / regional" is an **interpretation** and must be an
  explicit, editable table — not silently baked in.
- **H4 — Pie charts require a partition.** Theories are not mutually exclusive; papers
  span several. Either force a single dominant label (and disclose the confidence), or
  switch to a stacked bar / treemap which permits multi-label. A pie chart over
  overlapping categories is formally invalid.
- **H5 — Community-discussion data is the least freely available.** Altmetric is
  commercial. Expect to defer this class or substitute weak proxies, and mark it as
  lower-confidence in the weighting.
- **H6 — Self-citation and citation cartels** inflate counts; `influentialCitations`
  partially mitigates.

## 5. Open questions (need the user's decision)

- **Q1 — "k-index" definition.** The h-index family is an *author* or *venue* level
  metric; there is no standard paper-level k-index. Should author h-index be attributed
  down to their papers, or is a different measure intended?
- **Q2 — Edge semantics.** Confirmed as citations only? (Recommended: yes — it is the
  only relation that constitutes a genuine "pathway".)
- **Q3 — The remaining influence classes** the user has said they will supply.
- **Q4 — H4 resolution:** single dominant label, or multi-label with a non-pie chart?
- **Q5 — Corpus scale.** Candidate-set size N, and how many citation hops.

## 6. Consequences

- Adds one new external-data dependency class (bibliometrics) behind a Protocol; no
  heavy local model dependencies.
- Graph building becomes the project's first genuinely long-running job — requires
  resumability and caching as first-class concerns.
- The evaluation harness (L6) finally makes quantitative claims falsifiable.
- The trend engine (Branch A) is unaffected and remains the field-level measurement;
  this graph is the *query-level* instrument.
