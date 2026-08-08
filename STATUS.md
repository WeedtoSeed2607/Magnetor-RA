# Magnetor — Status & Roadmap
_Snapshot: 2026-07-30 · 192 tests green · ruff + mypy (strict) clean_

Domain-Aware Research Platform. Core principle: **Isolated Storage, Federated
Retrieval** — no document body/embedding ever crosses a domain directory.
Python 3.14, src layout. Core deps: httpx, numpy, scikit-learn; streamlit is an
optional `[dashboard]` extra. Code on GitHub: **WeedtoSeed2607/Magnetor-RA**
(branch `main`); not yet deployed to Streamlit Cloud.

---

## ✅ Built and working

| Area | What | CLI |
|---|---|---|
| Acquisition (§3–4) | 6 domains: qm/math (arXiv), neuro (PubMed) live; philosophy/anthro/history manual drop-folder. Watermark + cadence gates. | `acquire` |
| Retrieval (§5, §7.1) | Voyage embeddings (voyage-4-lite), numpy-cosine VectorIndex, CrossDomainRouter (top-K representative match). | `embed`, `query`, `status` |
| Branch B deep-dive (§7.2) | Anchor-Lock vs Field-Map, Semantic Scholar citation expansion, grounded-context payload (LLM-ready). | `deepdive` |
| Branch A trends (§6) | scikit-learn LDA (gensim infeasible on 3.14), volume-gated, drift + emerging terms, LOW SUPPORT flagging. | `trends` |
| Dashboard (§11) | Streamlit; Topic-Trend Banner, Primary Viewport, Frontier Feed, Evidence Graph panel; clickable links; password-gated search. | `dashboard` |
| Branch C graph (ADR-0006) | OpenAlex harvest + snowball expansion; in-degree + PageRank influence; bootstrap rank CIs + leakage; graphviz node map. | `harvest` |
| Branch C navigation (L5.1–L5.3) | In-graph search box; Personalised-PageRank highlighting; **two-way** k-best progression paths (roots ← origin → development) with "Next"; node detail panel; leg-coloured edges; rough-guide disclaimer. | `dashboard` |
| Branch C relations (L4) | Co-citation + bibliographic coupling, computed from reference lists already held — **no extra API calls**. Own layers in the graph doc, dashed/arrowless in the render (I2), **excluded from influence** (D4). On the Morality graph they add 199 pairs the citation backbone cannot express. | `harvest` |
| Branch C node inspection | Hide-pathway toggle; pick 2+ papers and get lineage / indirect / unconnected with the chain shown and highlighted; subgraph-only view; per-paper "shares references with" list. | `dashboard` |
| Branch C facets | Multi-label **approach** classification (empirical / formal / mechanistic / normative / conceptual / linguistic / unclassified) done at harvest time from title+abstract; only the label and its evidence terms persist. Drives **cross-facet discovery**: papers sharing your foundations that attack the problem differently. | `harvest`, `anchor` |
| Branch C coding instrument | Part III schema, append-only per-graph store, prefill that never asserts more than was checked, and **Krippendorff's alpha** per field with the manual's floors. The instrument, not the study. | `dashboard` |
| Branch C breadth check | Judges whether a query was too broad **from the graph** (pair coverage + fragmentation share, not the query string) and proposes narrower questions ranked by citation **cohesion** rather than word frequency. One click harvests a suggestion. | `dashboard` |
| Branch C anchored mode | Build a graph outward from **one paper** (DOI / OpenAlex id / link) instead of a question: backward through references, forward through citers. Implemented as a `WorksSource`, so the whole harvest pipeline is inherited unchanged. | `anchor` |
| Branch C lineage export | Download a traced lineage as a folder: reading order, CSV, BibTeX, provenance. Citations and links, never the PDFs. | `dashboard` |
| Branch C harvest launcher | Start a harvest from the page instead of a terminal. Spawns the CLI as a **background job** (I5 intact — never in the request cycle), polls an exit marker, tails the log. Off unless `MAGNETOR_ENABLE_HARVEST_UI=1`, which only `mag.cmd` sets, so the hosted app can never spawn a job. | `dashboard` |

**Recent (this session):** snowball expansion (leakage 96%→82%, pulls true field
origins — Shor/CSS/Deutsch — into the QEC graph); self-loop drop + flagging.
Then L5.1–L5.3 navigation (`pathways.py`): lexical seed, no new dependencies, all
computed server-side so the static renderer still works. Verified on the real QEC
graph — "surface code" traces back to *Perfect Quantum Error Correcting Code*
(1996), "fault tolerant threshold" to Aharonov–Ben-Or (1997). Also added the
§3/I4 isolation test the ADR required but nobody had written: a graph document
provably carries no abstract bodies.

## 📦 Current data state (`data/`)
- Records: **qm 130, math 130, neuro 134** — all embedded.
- Trends: `trends.json` for qm/math/neuro.
- One Evidence Graph harvested: **"quantum error correction surface code"** (232
  papers after 2 expansion rounds).

---

## 🔜 What's next / to be added

### Branch C — approved metric catalogue (skeleton has only 2 core metrics so far)
1. **Foundational-completeness metric** — "external works cited ≥2× still outside
   the set." Replaces raw leakage as the honest completeness signal (raw leakage
   overstates — see weaknesses). Cheap; do this first.
2. **Full weighted composite** (L3.1) — versioned profile file blending the below.
3. **Signal providers:** h-index (m-quotient / field-percentile / self-cite-
   stripped), S2 citation intent + context-depth, venue reach (author-country
   spread), scrutiny/integrity (retractions/PubPeer), applications (patents/
   trials, within-domain only).
4. **Influence typology** (theoretical/methodological/empirical/contrarian) +
   contrarian on a separate visual track.
5. **Boundary depth-convergence** diagnostic (kendall_tau on fixed sets — the
   primitive already exists in `robustness.py`).

### Dashboard rendering feedback (ADR-0006 §L5.1–L5.3) — ✅ BUILT
Decisions taken 2026-07-30, and what they cost:
- **Node detail panel** — full title / link / per-metric influence breakdown, via
  *selection* not click: `st.graphviz_chart` is a static image, and swapping to a
  JS graph component was declined to keep the hosted app dependency-free.
- **No abstract in the panel.** L5.1's build note claimed the abstract was already
  on the node; it is not, deliberately (§3/I4). Kept out, so the panel is
  identity + link + metrics. Costs the in-app reading pane, keeps the invariant
  and keeps abstract text out of the committed snapshot.
- **Search box (L5.2)** — lexical seed → Personalised PageRank. *Limitation:* the
  seed matches **titles only** (no abstracts stored), so synonyms are missed. The
  embedding seed remains the documented upgrade.
- **Progression paths (L5.3)** — **two-way** (`roots ← origin → development`), not
  the one-way forward walk the ADR assumed; endpoint is the open-ended walk.
  Traced legs are drawn thick and colour-coded (purple back, green forward) so a
  path is traceable by eye, which also settles L5.1(2)'s "what is a pathway".
- The floor applies to the **blended** step weight, not raw keyword relevance — a
  raw floor severs the roots leg, since foundational papers rarely match a keyword.

**Still open here:** relevance is title-only (embedding seed would fix); true
click-on-node needs the deferred interactive component.

### LLM-dependent (deferred — gateway undecided)
- Critic Agent §8, Pedagogical §9, Branch C **L6 position synthesis + pie chart**.
- Gateway: **Anthropic API** (server-side, cleanest) or **HF-router open models**
  behind a `Synthesizer` Protocol. Puter.js and "Claude from Hugging Face" both
  investigated and **ruled out** (browser-only / no Claude weights).

### Other
- Sentiment §6 (declined). Cross-domain dedup §10.
- **Deploy:** repo pushed; you deploy on share.streamlit.io — main file
  `streamlit_app.py`, secrets `MAGNETOR_VOYAGE_API_KEY` + `MAGNETOR_SEARCH_PASSWORD`.

---

## ⚠️ Known weaknesses (honest)
1. **No evaluation.** 168 tests prove *mechanics*, not *answer quality*. Highest-
   value gap: a ~30-query labelled relevance set — unlocks threshold calibration
   and metric validation. Everything below is downstream of this.
2. Anchor threshold (ADR-0004) is median-based → ~half of queries trip Anchor-
   Lock by construction; not label-calibrated (n=18).
3. Trend binning is fixed-width *time* → sparse/empty slices; quantile (equal-
   doc) binning would fix it.
4. LDA topic ids are not comparable across runs — `topic_trend_log.json`'s
   longitudinal framing is a trap for future-you (nothing reads it yet).
5. Raw boundary leakage overstates incompleteness (long tail of singleton refs);
   see foundational-completeness metric above.
6. Password gate is a speed bump (plain `==`, no rate-limit), not access control.
7. Snapshot staleness (`generated_at`) not surfaced in the dashboard.
8. **Anchored mode's seed always ranks first, tautologically.** The set is built
   from papers citing the seed, so its in-degree measures the gathering, not the
   literature. The ranking is only meaningful *among the other* papers. Flagged in
   the CLI and the UI; not fixable without a different normalisation.
9. **Facets are an unvalidated screening heuristic.** Lexicon matching, never
   measured against human coders, so the manual's reliability floor is unmet.
   54% of works come out `unclassified` — and the split is the point: 96.5% for
   papers with no abstract vs 43.2% for those with one, and OpenAlex holds no
   abstract for 20% of works. A third of that rate is absent data, not a bad
   classifier. The coding instrument exists partly to measure this.
10. **The verdict layer is a consistency check, not a measurement.** It reports
    whether one coder's judgement hangs together under II.9. Known gaps, all
    surfaced in the UI: the unit is the **paper**, but III.1 codes the *claim* and
    III.3's fields are per-*revision*; core/belt is **not pre-registered**, so
    Feyerabend's retrospective-relabelling objection stands; the pessimistic
    roll-up and the 10-year window are **stipulations of this implementation**,
    not the source's; and defeaters live in code rather than the versioned profile
    file II.4 asks for.
11. **The coding corpus is unbuilt and the manual unpiloted** — the source
    document says so itself (Part IV.1). What exists is the instrument and its
    alpha; the historical episode set of II.14 does not exist.
11. **In-graph search matches titles only** (L5.2). Abstracts are absent from graph
   artifacts by design (§3/I4), so a sub-query misses synonyms and any concept not
   named in the title — "decoder" finds nothing in the QEC graph even though the
   topic is present. Embedding the seed is the known fix; it needs a Voyage pass
   over graph nodes, which Branch C does not currently do.

## ❓ Open decisions awaiting you
- ~~L5.3 progression path: **endpoint** + **direction**~~ — resolved 2026-07-30:
  open-ended walk, **two-way** (roots *and* development), tagged edges highlighted.
- Build the **foundational-completeness metric** + soften the dashboard leakage
  warning? (offered, not yet done)
- **LLM gateway** choice before any synthesis/critic/pedagogical layer.

## ▶️ Quick run reference (from project root, venv built)
```
.\.venv\Scripts\pytest.exe -q          # tests
.\mag.cmd status all                    # data state
.\mag.cmd harvest "your question" --expand 2   # build an Evidence Graph
.\mag.cmd dashboard                     # launch the UI
```
Green gate before any commit: `ruff check src tests` · `mypy` · `pytest`.
See `docs/decisions/` (ADR-0001…0006) for the reasoning behind each choice.
