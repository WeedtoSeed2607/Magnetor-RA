# Magnetor — Status & Roadmap
_Snapshot: 2026-07-25 · 168 tests green · ruff + mypy (strict) clean_

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

**Recent (this session):** snowball expansion (leakage 96%→82%, pulls true field
origins — Shor/CSS/Deutsch — into the QEC graph); self-loop drop + flagging.

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

### Dashboard rendering feedback (ADR-0006 §L5.1–L5.3, deferred)
- **Node pop-up:** click → full title / abstract / link / influence breakdown.
  Needs an *interactive* graph component (current graphviz render is static).
- Edge visibility / colour; "rough guide" disclaimer.
- **Query-driven pathway highlighting** (L5.2): Personalised PageRank seeded by a
  cheap lexical (or embedding) relevance match — reuses existing PageRank.
- **Traceable progression paths, k-best** (L5.3): origin = *earliest paper most
  relevant to the keyword*; α slider (keyword ↔ influence). **PENDING your
  confirm:** endpoint (open-ended vs newest frontier) + direction (forward vs
  backward-to-roots).

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

## ❓ Open decisions awaiting you
- L5.3 progression path: **endpoint** + **direction** (above).
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
