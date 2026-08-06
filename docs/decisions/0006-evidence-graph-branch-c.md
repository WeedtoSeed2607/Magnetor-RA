# ADR-0006 — Branch C: The Evidence Graph (PROPOSAL / DRAFT)

Status: **partially implemented.** The walking skeleton of §10 steps 1–2 is built
(L1 harvest, L3.0 in-degree + PageRank, L4 graph, L5 render, L5.1–L5.3
navigation, L8 bootstrap ranks). The weighting **classes and their hierarchy
remain deferred to the operator** (§9 Open Decisions, D4) — no multi-signal
composite exists, so "influence" is still the single PageRank-percentile metric.
L2 signal registry, L6 positions and L7 audit export are not built.

Supersedes the previously-discussed "point-7 Outlook board" (rigorous definition
→ etymology → history → philosophical roots). That design is **withdrawn**: its
etymology/history/philosophy sections require generative world-knowledge with no
verifiable source in the corpus, and therefore hallucinate in a way no amount of
prompting removes. Conceptual/philosophical framing returns later (§8) as
*rhetorical questions posed by* a grounded summary, never as asserted fact.

---

## 1. Governing principle

**Provenance is the deliverable; ranking is a navigational aid.**

Operator's framing (verbatim intent): *"Even if the metrics don't turn out to be
accurate for all cases, at least there is a traceable pathway of research
papers."* This inverts the usual bibliometric priority and is the correct call —
it makes the artifact falsifiable and useful even when the weighting is wrong.

Three consequences, binding on all downstream design:

- **C1 — No number without its inputs.** Every displayed score decomposes into
  per-class contributions, each traceable to a source record (provider, field,
  value, fetch timestamp, URL).
- **C2 — No LLM-authored quantities.** Percentages, counts, and rankings are
  computed from stored assignments. A model may *label* and *phrase*; it may
  never *state a magnitude*.
- **C3 — Absence is displayed, not imputed.** A missing signal renders as
  "unavailable", never as zero, and never silently drops the paper.

---

## 2. Position in the system

- Branch A (§6) — Topic-Trend Tracking — statistical, LLM-free.
- Branch B (§7) — Technical Deep-Dive — Anchor-Lock / Field-Map.
- **Branch C (new) — Evidence Graph** — query-scoped influence map + grounded
  position summary.

Branch C reuses Branch B's retrieval and Branch B's `render_grounded_context()`
seam; it does not replace either branch.

---

## 3. Isolation invariant — reconciliation

Branch C is **topic-centric**, while Magnetor's storage is **domain-siloed**.
The reconciliation:

- Harvested papers are still written into their **own domain silo**. No document
  body or embedding crosses a domain directory. Unchanged invariant.
- The graph is a **derived artifact** holding *identifiers, metrics, and edges
  only* — never abstracts, never vectors. It may therefore span domains, exactly
  as `routing_log.jsonl` and `topic_trend_log.json` already do (Spec §10).
- Federation happens at the *answer* layer, which is the design's whole point.

Verdict: Branch C is compatible with "Isolated Storage, Federated Retrieval"
provided the graph artifact stores no bodies. This is a hard constraint, to be
enforced by a test, not a convention.

---

## 4. Data sources (honest availability assessment)

| Class | Best source | Availability | Confidence |
|---|---|---|---|
| Citation edges (for subgraph) | **OpenAlex** (`referenced_works`) | Free, no key, ~250M works | High |
| Substantive-citation signal | S2 `influentialCitationCount`, citation `intents` | Free | Medium-High |
| Author h-index | OpenAlex authors / S2 `hIndex` | Free | Medium-High |
| Institution credibility | OpenAlex institutions (country, type, metrics) | Free | Medium |
| Venue reach class (§4.1) | Derived from author-institution country spread | Free (computed) | Medium |
| Attention (altmetrics) | Paperbuzz / Crossref Event Data (open); Altmetric, PlumX (paid) | Open path exists; social coverage degraded since 2023 | **Low** |
| **Scrutiny / integrity** | Crossref + Retraction Watch; errata; PubPeer | Free | Medium |

### 4.1 Operationalising "international / national / regional"

Per operator clarification these are **independent additive credibility signals,
not an ordinal ranking** — presence in *any* class adds weight; classes are never
compared against each other and regional presence is never a penalty.

Something must still *classify* a venue/institution. This ADR proposes a
**descriptive** basis — **geographic reach**, computed from the country spread of
author institutions (OpenAlex carries country codes) — rather than an
**evaluative** one (prestige/impact factor). Reach is measurable and does not
encode reputational hierarchy. *Operator confirmed: reach, not reputation.*

**Correction — reach is NOT a reliability proxy.** The operator's stated rationale
("reach ... is a decent generalised proxy of how reliable a paper is") does not
hold and this ADR declines to encode it. Reach measures *distribution*;
reliability is an *epistemic* property. They dissociate hard: predatory
open-access venues maximise accessibility while minimising rigour; paywalled
specialist journals restrict reach while maintaining it. Treating reach as
reliability would import an availability bias into the credibility score. Reach
is therefore retained **only as a descriptive class flag**, never as a
reliability weight. Reliability is carried by scrutiny signals (§4.2) instead.

**Retained and promoted — niche-venue uptake as a *directional* signal.** The
operator's second rationale is sound and is a *different* metric: presence in the
specialist journals *of the theory's own niche* signals where the field is
heading. This is implemented as **venue-uptake trajectory** — the set of
specialist venues publishing this line of work, and whether that set is widening
or narrowing over time. It is a **direction** indicator, not a quality one, and
is rendered separately from influence magnitude.

### 4.2 Community discussion — expert-gated only

Per operator directive, only forums that are **expert-gated and formal in
register** qualify; anything general-audience or noisy is discarded.

| Source | Status | Coverage |
|---|---|---|
| **PubPeer** | post-publication peer review; formal | cross-domain, thin |
| **MathOverflow / TCS SE** | research-level, reputation-gated (Stack Exchange API, free) | math/CS only |
| Crossref + Retraction Watch | retractions, errata, corrections | cross-domain |
| ~~Reddit / Hacker News~~ | **discarded** — not expert-gated, CS-skewed | — |
| ~~General social altmetrics~~ | **discarded** — popularity proxy (contradicts L3.0); coverage degraded since the 2023 platform API lockdowns | — |

**Known limitation:** expert-forum coverage is severely domain-skewed. MathOverflow
serves mathematics well; most fields have no equivalent. This class will be
**near-empty for several domains**, so it cannot carry meaningful weight in a
cross-domain composite and must be treated as *bonus evidence where present*,
never as a required input. Its absence must not depress a paper's score.

**Split by function** (the class is not one thing):
- **Attention** (mentions, shares) → popularity proxy → **deprioritised**, per L3.0.
- **Scrutiny** (PubPeer critique, errata, retractions, replication) → *credibility*
  → **retained as an integrity flag on the node**, not a score input. A retracted
  or contested paper sitting in a lineage is exactly what a researcher must see.

**OpenAlex is the recommended backbone.** Decisive advantage: it returns
`referenced_works` inline, so the citation graph is built from the harvest itself
rather than N follow-up API calls per paper.

---

## 5. Pipeline (layered)

### L0 — Query specification
A *specific* enquiry sentence (operator's constraint — specificity is what makes
the "which question do these theories answer?" framing meaningful). Persisted as
a `QuerySpec` with a stable hash; every downstream artifact is keyed by it.

### L1 — Topic Harvest *(new acquisition mode)*
Today's acquisition is **date-watermarked and category-bounded** — a recency
feed. Branch C needs a **query-driven deep backfill**. New `TopicSource`
Protocol; `OpenAlexHarvester` as first implementation (cursor pagination, no
date floor). Output: papers + citation edges + venue/author metadata.
This is the prerequisite for the "bigger dataset" in the operator's point 1.

### L2 — Signal ingestion ("classes")
A `SignalProvider` Protocol, one implementation per class, in a **registry** so
classes can be added without touching the pipeline (the operator's class list is
deferred — the architecture must not assume it). Each provider returns per-paper
`SignalValue{value, provenance}`.

### L3 — Influence definition, normalisation, weighting

**L3.0 — Influence is query-relative, not global (operator directive).**
"The key influence comes from how much it influenced the particular theory or
question, not from how much people like it." Global citation count is therefore
**demoted to a low-weight credibility signal** and is *not* the influence metric.
Influence is computed **within the harvested subgraph**:

- **In-subgraph in-degree ("repetition")** — citations received *from papers in
  the topic set*. A paper with 10,000 global citations but 2 in-set citations is
  not influential to this question. Falls out of `referenced_works`; no extra
  API calls. *This is the computable form of the single most reliable human
  heuristic: the same paper appearing again and again across the field's
  introductions and reference lists.*
- **Subgraph centrality** (PageRank / betweenness) — structural importance within
  this lineage.
- **Substantive-citation filter** — S2 `influentialCitationCount`; and citation
  `intents`, weighting *method*/*result* citations above *background* (which are
  frequently ceremonial).
- **Citation-context depth** — S2 exposes citation `contexts` (the sentences
  around each citation). A work discussed at length (design described, findings
  recounted, positioned as the origin of a line of work) is being treated as
  load-bearing; a work named once inside a list is not. Operationalised as
  mean context length / mentions-per-citing-paper. *Novel, and directly tracks
  "how much attention authors devote to it".*
- **Recency-weighted in-degree** — in-set citations arriving from *recent*
  papers weighted higher, so preprints and new work that current studies keep
  framing their questions around surface before they accumulate raw counts.
  Directly counters R2 age bias.
- **Review-article endorsement** — reviews describe a field's structure and
  emphasise the work it considers foundational. Detect reviews (OpenAlex/S2
  type) and treat *their* reference lists as a weighted signal — **flagged as
  perspective-biased**, since reviews reflect their authors' emphases.

**L3.0a — Influence is a typology, not a scalar.** A single composite number
would erase the distinction between kinds of influence. Nodes therefore carry an
**influence type** alongside magnitude:

| Type | Evidence signature |
|---|---|
| **Theoretical** | later work adopts its framework/vocabulary; high background+method intent |
| **Methodological** | its instrument/method is reused; method-intent citations dominate |
| **Empirical** | supplies evidence that settles a question; result-intent citations |
| **Contrarian** | challenges assumptions; generates sustained follow-up, replication, dispute |

This matters for rendering (node *shape/colour* = type, *size* = magnitude) and
prevents the contrarian case from being misread as consensus — a paper cited
heavily *because it is disputed* is not the same as one cited because it is
adopted.

Consequence, and it is the correct one: **the same paper scores differently for
different queries.** Influence is a property of the paper *with respect to the
enquiry*, not an intrinsic property of the paper. This also substantially
defuses R2 (age bias), since in-set degree is relative to the topic cohort.

**L3.1 — Normalisation.** Remaining raw signals are scale-incommensurable
(citation counts are power-law; h-index is bounded; presence flags are binary).
Naive weighted summation is invalid.

- Normalise by **percentile rank within the harvested set** (distribution-free).
- Citation counts additionally **age-normalised** within publication-year cohort
  (see §7 Risk R2).
- Weights live in a **versioned profile file, not code**, so competing
  hierarchies can be swapped and compared.
- The composite retains its **per-class decomposition** (C1).

### L4 — Graph construction
- **Nodes** = papers. **Size** = influence magnitude; **shape/colour** =
  influence *type* (L3.0a); domain available as an alternate encoding.
- **Edges** = typed, each independently filterable:
  - `cites` — direct citation (the lineage spine).
  - **`co_cited`** — two papers repeatedly cited *together* by later work.
    Co-citation indicates the field treats them as related, and it is how
    shared intellectual antecedents surface without either citing the other.
  - **`biblio_coupled`** — two papers sharing a substantial set of *references*.
    Reveals studies built on the same foundation (same instrument, same
    framework) even when they never cite each other.
  - `co_author`, `same_venue`, `semantic_similarity` (existing embeddings).
- Co-citation and bibliographic coupling are the computable form of "compare
  several studies side by side and see which earlier work they all lean on."
  Both are derivable from data already fetched — **no additional API calls**.

> **BUILT 2026-07-30 — `relations.py`.** Measured before building: on a 144-paper
> sample, direct citation joins **2.2%** of pairs while >=3 shared *external*
> references join **19.8%**. So the boundary-leakage figure the dashboard reports
> as a weakness is simultaneously the substrate for the relations the backbone
> cannot express. On the shipped defaults (>=3 shared refs, >=2 co-citations,
> top-6 per node for R4 hairball control) the "Defining Morality" graph gains
> **199 pairs that no citation joins**, taking pair coverage 11.9% -> 23.2%. The
> pruning is why this is ~2x rather than the ~9x the unpruned sample suggests.
> Two constraints held: the layers are stored and drawn separately from `cites`
> (I2 — dashed, arrowless), and they are **excluded from influence** (D4 gates
> the metric catalogue), so they change what is visible and never what is scored.
> *Known contaminant:* OpenAlex duplicate records for one paper appear as a
> strong self-pair and inflate both layers; dedup is outstanding.
- Persisted as `graphs/<query_hash>.json` — IDs, metrics, edges, provenance.

### L5 — Rendering
Interactive network in the dashboard; click a node → paper card with the
existing `paper_url()` links; filters by class, year, domain, min-influence.
Node budget enforced (§7 Risk R4).

#### L5.1 — Rendering feedback (operator, 2026-07-23; **BUILT 2026-07-30**)

Captured against the skeleton graph (static Graphviz DOT via `st.graphviz_chart`).

**Implemented as** `dashboard_data.node_detail` + `graph_dot`'s `traced` /
`highlighted` / `anchor` arguments, with two operator decisions on 2026-07-30:

- **(1) is delivered by *selection*, not click.** The renderer stays static — the
  JS-component swap this section warns about was declined, so a paper is chosen
  from a list and its detail panel renders beside the graph. Same information,
  no new dependency, no CDN at runtime.
- **The abstract is NOT shown, and this section's build note was wrong about it.**
  "Abstract text is already harvested and persisted on the node" holds only for
  `HarvestedPaper` in memory; `build_graph_document` deliberately omits it,
  because §3/I4 forbids bodies in the artifact. Kept omitted. The panel therefore
  carries identity, the outbound link, and the full per-metric breakdown (C1).
  §3's "enforced by a test, not a convention" is now actually true —
  `test_document_stores_no_abstract_bodies` was missing and has been added.
- **(2) is settled by L5.3's traced path:** the traced legs *are* the pathway, so
  "unique pathway" needed no separate definition. Untraced edges darkened to
  `#8a8a8a`; traced legs drawn at `penwidth=3` in purple (roots) / green
  (development). **(3)** ships as `GRAPH_DISCLAIMER`, rendered under every graph.

1. **Node labels fit inside the node, ellipsis on overflow.** Truncate the title
   to the node width with a trailing "…". Clicking a node opens a detail box:
   **full title · abstract · link to the paper · influence analysis**
   (per-metric breakdown, not just the composite).
   - *Build note:* the skeleton's `st.graphviz_chart` renders a **static** image —
     no native click-to-inspect. Click-to-expand requires an interactive graph
     component (a JS network library in a Streamlit custom component), i.e. a
     rendering-engine swap, not a tweak. Abstract text is already harvested and
     persisted on the node, so the data for the box exists.

2. **Make pathways clearly visible** — a darker/striking edge colour, or a
   distinct colour per unique pathway (current skeleton draws uniform light-grey
   edges).
   - *Open question to settle before build:* "unique pathway" is ambiguous in a
     citation DAG because paths share edges. Needs a chosen definition — e.g.
     colour by root-ancestor lineage, by weakly-connected component, or
     highlight-a-single-traced-path on hover — before colour-per-pathway is
     well-defined. A single darker uniform edge colour is unambiguous and cheap
     if per-pathway proves not worth it.

3. **Disclaimer beside the graph:** influence levels are a **guide for a rough
   idea only**, relative to the number of papers harvested (percentile-normalised
   *within the set*, not absolute or authoritative). Consistent with the ADR's
   own stance (L8 robustness, §8 researcher validation) and the "even if the
   metrics aren't accurate for every case, at least there's a traceable pathway"
   principle. The existing boundary-leakage warning is a related honesty cue.

#### L5.2 — Query-driven pathway highlighting (operator, 2026-07-23; **BUILT 2026-07-30**)

**Implemented** in `pathways.py` as `highlight()` over `personalised_pagerank()`.
The cheap **lexical** seed was taken, as this section recommends, so no Voyage
pass and no key are involved. **Honest limitation discovered in build:** the seed
can only match **titles** — this section assumed "title/abstract", but abstracts
are absent from the artifact by §3/I4 — so synonym and related-concept matching is
weaker than described here. The embedding seed remains the stated upgrade path.
Dangling teleport mass returns to the seed rather than uniformly, so relevance
cannot leak away from the query.

**Ask:** a *second* search box, scoped to an already-loaded graph, that takes a
sub-query (operator example: "Mizar System Applications") and **highlights the
pathway of nodes most likely related to it**, using the existing engine
principles. This is distinct from the harvest query that *built* the graph — it
illuminates a relevant sub-lineage *within* it, for focused analysis.

**Principled mechanism (reuses what's already here):** seed a **Personalised
PageRank / random-walk-with-restart** with a per-node **semantic-relevance
vector** (Voyage cosine between the sub-query and each node's title+abstract).
High-PPR nodes + the citation edges among them are the highlighted pathway. This
composes the two engines already in the codebase — the hand-rolled PageRank
(L3.0) and the Voyage embedding similarity (Branch B) — rather than inventing a
new one.

**Separate the relevance signal from the ranking engine (operator insight,
2026-07-23).** The path *ranking/propagation* reuses the existing engine wholesale
— Personalised PageRank is the L3.0 PageRank with a seeded (non-uniform) teleport
vector; edge highlighting falls out of node scores (an edge is on the relevant
path when both endpoints score high). Two important corollaries:
- **Influence ranking alone is query-blind and cannot do this by itself.**
  In-degree/PageRank give one global centrality per node with no notion of
  "aboutness"; using them to highlight "paths related to *Mizar*" would return
  the globally most-influential papers and the *same* highlight for every
  sub-query. The query must enter through a *content* signal — structure can't
  manufacture one.
- **But the seed can be cheap.** It does NOT require Voyage. A **lexical match**
  (query terms in title/abstract) is enough to seed the walk — free, no API, no
  key — and is a fine v1. Embeddings only upgrade the seed (synonyms, related
  concepts), so they become an optional enhancement, not a prerequisite. This
  removes the main build cost previously flagged here.

**Query-agnostic edge importance (no seed needed):** to merely emphasise the
load-bearing connectors, weight edge `(u→v)` by `PageRank(u)·PageRank(v)` (or the
cited endpoint's PageRank), or by edge betweenness — pure reuse of the ranking
system, no query involved. A clean standalone win.

**Other considerations:**
- Branch C stores abstracts but does not embed them (embedding lives in the
  per-domain Branch-B pipeline), so the *embedding* seed variant would need a new
  Voyage pass; the *lexical* seed variant needs nothing new.

#### L5.3 — Traceable progression paths, k-best (operator, 2026-07-23; **BUILT 2026-07-30**)

Refines L5.2 from a highlighted node *set* to a single ordered **path** with
ranked alternatives — the operator's clarified intent.

**Ask:** on the graph from the initial harvest question, the operator types a
keyword; the tool returns a **single traceable path** — origin → node → node → …
via connectors — that (a) starts from the **origin** (foundational root of the
question within available data), (b) follows the **most probable progression** of
nodes through citation connectors, **biased to match the keyword**, and (c)
offers a **"Next"** control stepping to the next-most-likely progression.

**Algorithm — k-best maximum-probability path in the citation DAG** (NOT PPR;
PPR yields a weighted set, this needs an ordered walk):
- **Direction:** edges are citing→cited (new→old); a *progression* runs
  origin→later, so the walk traverses **cited→citing (reversed)** — foundational
  work up to what builds on it.
- **Step weight** `w(u→v)` = keyword relevance of `v` (lexical, free; embeddings
  optional) × `v`'s influence (reuses L3.0 PageRank). Normalise outgoing → probs.
- **Most-probable path** = max product of step probs = heaviest path in a DAG →
  linear-time DP. **"Next"** = **k-best paths** (Yen / DP variant).
- **Reuses** influence ranking + a cheap keyword seed; **new** = the path-search
  DP (small). Confirms the operator's "same ranking system" intuition: the
  ranking feeds the *weights*; a path algorithm is still needed on top.

**Critical honesty flag — "origin" = origin *within the harvested slice*.** With
~96% boundary leakage (L8), the true historical origin is almost certainly
*outside* the set (roots are older; keyword search skews recent). The path's
"origin" is "the earliest/most-foundational node this harvest captured," not the
field's real origin. **This feature is only as trustworthy as harvest
completeness toward the roots — it depends on the deferred snowball/boundary
expansion**, and must be labelled as such in the UI.

**Caveats:** citation graph is *mostly* a DAG but not guaranteed (self-loops
already removed; genuine cycles rare — the DP needs a cycle-guard). Descriptive,
not authoritative (reuse L5.1(3) disclaimer).

**Resolved (operator, 2026-07-23):**
1. **Origin = earliest paper among those most relevant to the keyword**
   (relevance filter → chronological earliest). *Operator clarification
   (2026-07-23) supersedes the clicked "strongest keyword-matching": "the
   earliest possible paper most relevant in the query, not the earliest paper
   within the graph."*
   - **Two-factor selection:** (a) relevance filter — candidates are papers above
     a relevance threshold to the keyword (**the threshold is a knob**: too low
     and a marginally-relevant ancient paper wins the origin); (b) among
     candidates, take the **earliest published**.
   - *Consequence (honest, and it returns in full):* "earliest relevant *within
     the harvested slice*" is a direct **completeness claim**, and with ~96%
     leakage the true earliest-relevant paper is very likely *outside* the set.
     So this origin is trustworthy only after snowball root-expansion; the UI
     must label it **"earliest in available data,"** not "the origin."
   - *Consequence (positive):* an early origin yields a **long, rich forward
     progression** — better aligned with the feature's purpose (trace how the
     work developed) than "strongest," which risked a stub when the anchor was
     recent.
2. **Step weighting = tunable blend:** `w(v) = α·keyword_relevance(v) +
   (1-α)·influence(v)`, with **α exposed as a slider** (keyword ↔ influence);
   default α mid. Path probability = normalised product along the walk.
3. **Endpoint — CONFIRMED (operator, 2026-07-30):** open-ended walk, stopping
   when no next step clears the floor or at max depth. The walk may not stop
   *voluntarily*: every step multiplies by p<1, so an opt-out would make the empty
   path always win. Rejected alternative: run to the newest relevant frontier.

**Direction — RESOLVED, and the assumption was wrong (operator, 2026-07-30):**
"progression" is **two-way**, not forward-only. One path runs
`roots ← origin → development`, so it shows both what the origin was built on and
what grew out of it. Implemented as two k-best walks from the anchor — `roots`
follows stored citing→cited edges, `development` follows them reversed — joined
into a single chain, oldest first. Each step carries its leg tag, and the renderer
draws tagged edges thick and leg-coloured (purple back / green forward) so the
path can also be traced by eye, per the operator's requirement.

**One further deviation, forced by the two-way requirement:** the floor is applied
to the **blended** weight `w(v)`, not to raw keyword relevance as written above. A
raw relevance floor severs the roots leg immediately — foundational papers are
general and rarely contain the keyword (verified: on the QEC graph, tracing
"decoders improved" reaches its 1996 root only because influence carries that
leg). Lowering α lengthens the roots leg.

Status: **built** in `pathways.py`; k-best DP memoised on `(node, depth)`, which
doubles as the cycle guard this section asks for. Verified against the real
harvested QEC graph: "surface code" anchors on *Perfect Quantum Error Correcting
Code* (1996), "fault tolerant threshold" on Aharonov–Ben-Or (1997) — i.e. the
origin rule does surface genuine field roots once snowball expansion has run.
- **"Pathway" still needs a definition** — same open question as L5.1(2): a
  single traced path, the induced subgraph of top-relevance nodes, or the
  PPR-weighted lineage. PPR naturally yields a *weighted node set*, so
  "highlight = recolour nodes/edges by PPR score" is the least-ambiguous form.
- **Compatible with the current static renderer** (unlike L5.1(1)'s click-box):
  the highlight is *computed server-side on submit* and the DOT re-emitted with
  those nodes/edges recoloured — no interactive graph component required.
- Keep it **descriptive, not authoritative** — reuse the L5.1(3) disclaimer;
  "most likely related" is a ranked guide, not ground truth.

### L6 — Position synthesis + composition chart
1. Cluster paper embeddings (existing `VectorIndex`) → candidate positions.
2. A model **labels** clusters and **assigns** papers, fed the grounded context.
3. **Proportions are computed by counting stored assignments** (C2).
4. Every slice drills down to its paper list. Slice `n` always shown; thin
   slices flagged, reusing the Branch-A `LOW SUPPORT` convention.

### L7 — Audit layer (cross-cutting)
An export that reproduces every displayed number from stored records. This is
what makes researcher validation (§8) meaningful rather than impressionistic.

---

### L3.2 — The four influence criteria, and which are tractable

The operator defines influence as: *outreach · consistency amongst academia ·
internal consistency within the theory · applications · "much more."* These are
**not equally computable**, and the ADR states so rather than implying uniform
rigour:

| Criterion | Tractability | Implementation |
|---|---|---|
| **Outreach** | **High** | in-subgraph degree, centrality, recency-weighted degree (L3.0) |
| **Applications** | **Medium** | patents (Lens.org, free w/ login), clinical trials, software/data citations. Coverage uneven by field; policy-document tracking is paid-only |
| **Academic consistency / convergence** | **Medium** | co-citation clustering + citation-intent agreement: does later work converge on this path vs. a competitor? Measurable as cluster dominance over time |
| **Internal theoretical consistency** | **LOW — flagged** | see below |

**Internal consistency is not bibliometrically computable.** Whether a theory is
*logically* self-consistent is a property of its propositional content, not of
its citation metadata. Nothing in OpenAlex, S2, or Crossref encodes it. The only
routes are (a) an LLM reading full texts and judging coherence — unreliable, and
precisely the fabrication-prone register we removed from the design, or (b)
formal argument extraction, which is an unsolved research problem.

**Decision:** internal consistency is **excluded from the composite score** in
v1. Presenting a computed "consistency" number would be false precision of
exactly the kind this architecture exists to avoid. The graph instead surfaces
its *observable shadow* — disputes, replications, corrections, contrarian
citation patterns (L3.0a) — and leaves the coherence judgement to the reader.
Revisit only if researcher validation (§8) shows a tractable proxy.

---

### L3.3 — Resolved operator decisions

**D1 — Citation contexts fetched for ALL harvested papers** (not a two-pass
top-N). Operator: *"Time is not an influential factor here since reliability and
effectiveness of the framework is much more important."* Correctness over
latency, accepted.

> **Consequence (must be designed for, not discovered):** at S2's unauthenticated
> rate limit this is minutes-to-tens-of-minutes per query. **The harvest therefore
> cannot run inside a dashboard request cycle** — it would block the UI and exceed
> hosting timeouts. Harvest becomes an **offline CLI batch job** writing a
> persisted graph; the dashboard only *reads* it. This matches the existing
> `acquire`/`embed`/`trends` pattern exactly, so it costs no architectural
> novelty. Two mitigations are mandatory: a **local context cache** (contexts are
> near-static; never re-fetch) and use of `MAGNETOR_S2_API_KEY` for the higher
> rate tier.

**D2 — Contrarian papers occupy a separate visual track** with distinct
shape/colour, not merged into the consensus ranking (L3.0a).

**D3 — Applications are normalised WITHIN domain only.** A QM paper and a
neuroscience paper have incomparable application surfaces (patents vs. trials);
cross-domain comparison of this class is prohibited.

**D4 — Metric set requires operator sign-off before implementation.** Operator:
*"Consult with me the metrics used before implementing."* The catalogue is
therefore a **gated artifact** — no scoring code is written until each metric is
approved. Full methodology deferred pending empirical testing.

**D5 — Author metric = h-index** (operator confirmed; the literal bibliometric
"K-index" is Hall's satirical Kardashian index — social-followers ÷ citations —
and is rejected as a popularity measure contradicting L3.0). h-index is used
**only in corrected form**: **m-quotient** (h ÷ academic age) for career-length
bias, **field-percentile** rather than raw value for field dependence, and
**self-citation-stripped h** reported as a gaming delta.

---

### L8 — Robustness layer (the evaluation the system currently lacks)

**Governing insight:** the output is a *ranked, sized graph*, so absolute score
accuracy is rarely the question. What must be demonstrated is that the **top-N
set and ordering are stable under perturbation of every arbitrary choice**.
Every weakness below therefore reduces to one measurable quantity: **rank
correlation under perturbation** (Kendall τ / Spearman ρ; top-N Jaccard).

| Weakness | Method | Statistic | Action on failure |
|---|---|---|---|
| Harvest completeness | **capture–recapture** (two independent harvests; Lincoln–Petersen N≈ab/m) + saturation curve + boundary-leakage rate | est. coverage % | widen harvest; display coverage |
| Graph boundary | **convergence by iteration** — PageRank at snowball depth k vs k+1 | τ(k, k+1) | expand another round |
| Window length | **sensitivity sweep** (1/2/3/5y) | τ across sweep | expose as user parameter |
| Weight profile | sweep competing profiles | τ across profiles | report as a result surface, not a point |
| Citation intent | **human-labelled sample** (~100), measured against **inter-annotator agreement as ceiling** | Cohen κ | drop intent weighting |
| `influentialCitationCount` | criterion validity vs human labels | correlation | demote to tiebreaker |
| **Context depth** (unvalidated) | **confound test** vs venue/parser | partial correlation | **delete the metric** |
| h-index | m-quotient, field percentile, self-citation delta | Δ vs raw | corrected form only |
| Cluster stability | **consensus clustering** across refits | ARI / co-assignment | suppress unstable clusters |
| Author/institution ID | sample-and-audit; ORCID as truth where present | error rate | flag affected nodes |
| Venue reach | confound test vs author count | partial correlation | size-normalise |
| Scrutiny coverage | per-domain coverage rate | % with signal | mark class non-comparable |

**Bootstrap ranks (cross-cutting).** Jackknife-resample the graph to produce a
*distribution* of each paper's rank; publish **"rank 3 (95% CI 2–7)"** rather
than a bare "3." Papers whose CI spans a wide band are flagged unstable, reusing
the Branch-A `LOW SUPPORT` convention.

**Free gold standards.** Review-article reference lists are expert-curated
"papers that matter" — usable as a **zero-cost partial gold standard for harvest
recall**, before any human labelling. Researcher validation (§8) supplies the rest.

**Thresholds are deliberately unset** — to be calibrated during researcher
testing. Inventing cutoffs and presenting them as established would be exactly
the false precision this ADR exists to prevent.

#### L8.1 — Sample sizes (tiered; operator-agreed)

n=30 derives from the CLT rule of thumb for sampling distributions of a *mean*
and does not transfer uniformly to these checks. Sampling is therefore **tiered**:

- **Tier 1 — screen at n=30.** Cheap, and decisive only in the *negative*: a
  metric that fails at n=30 is killed immediately. A metric that passes is
  **not** thereby validated. Adequate margin at n=30 is roughly ±18% on a
  proportion — enough to detect "broken," not "good."
- **Tier 2 — confirm survivors at larger n, stratified.** Uniform sampling is
  prohibited for citation intent: background citations dominate (~60–70%), so
  n=30 yields ~3–5 *result*-intent items and per-class precision is
  inestimable. Sample **per class**.
- **Tier 3 — computational checks always run at full strength.** Bootstrap
  ≥1000 resamples; sensitivity sweeps exhaustive. These cost only CPU, so
  rationing them buys nothing.

**Power note:** confound tests are underpowered at n=30 (detecting r≈0.3 at 80%
power needs n≈85). An undetected confound is a shipped confound — these belong
in Tier 2 regardless of screening outcome.

**Addendum (2026-07-30) — starting a harvest from the dashboard.** The operator
asked to launch harvests from the page rather than a terminal. This does *not*
reopen D1/I5: the page **spawns** the CLI as a detached background job and then
only reads files (log, exit marker, then the graph), so nothing long-running ever
occupies a request cycle. Implemented in `harvest_jobs.py`, gated by
`MAGNETOR_ENABLE_HARVEST_UI` — default-deny, set only by the local `mag.cmd`
launcher and never by `streamlit_app.py`, so a hosted deployment cannot spawn
processes or spend quota from a public URL. Completion is read from an exit
marker written by a wrapper process, not inferred from the artifact and not from
a PID probe (`os.kill(pid, 0)` terminates the target on Windows rather than
testing it).

---

## 6. Module map

**New:** `harvest.py` (L1) · `signals/` package + registry (L2) ·
`scoring.py` (L3) · `graph.py` (L4) · `positions.py` (L6) · dashboard panels (L5)

**Reused unchanged:** `_http`, `DomainStore`, `Paper`, `VectorIndex`,
`CrossDomainRouter`, `SemanticScholarClient`, `render_grounded_context()`,
`dashboard_data` helpers, the Protocol-boundary idiom, the green gate.

---

## 7. Risks and weaknesses (stated before build, not after)

- **R1 — [SUBSTANTIALLY RETIRED after operator clarification].** The original
  concern — that tiering encodes prestige hierarchies and penalises regional,
  non-English and Global South venues — assumed an *ordinal ranking*. The design
  is **additive and independent** (§4.1): presence adds, nothing is compared,
  nothing is penalised. The equity failure mode does not arise. *Residual risk:*
  the classifier itself must stay descriptive (reach) rather than evaluative
  (prestige), or the problem re-enters through the back door.
- **R2 — [LARGELY MITIGATED by L3.0].** Age bias was severe when influence meant
  global citation count. Query-scoped in-subgraph degree measures influence
  relative to the topic cohort, which greatly reduces it. *Residual:* recent
  papers still accrue fewer in-set citations, so retain an age-normalised
  variant as a UI toggle.
- **R3 — Composite scores conceal disagreement.** Two papers can score alike for
  opposite reasons. Mitigated by C1 decomposition, not removable.
- **R4 — Graph hairball.** A broad keyword yields thousands of nodes; force-
  directed layout degenerates and browsers stall. Cap rendered nodes (~200–500),
  prune edges, use progressive disclosure. The *stored* graph stays complete.
- **R5 — A pie chart asserts mutual exclusivity that theories do not have.**
  Papers frequently span positions; a pie forces one label and sums to 100%.
  Resolution: assign one **primary** position (drives the pie, auditable) plus
  **secondary tags**, and surface the multi-position count beside the chart.
- **R6 — Community-discussion data may be unobtainable at acceptable quality.**
  Treat as optional and explicitly partial; never let it silently move rankings.
- **R7 — Validation needs a rubric defined in advance.** Researcher testing (§8)
  is only falsifiable if the expected pathways are written down *before* the test.
  Otherwise it collects impressions. This is the same unresolved gap as the
  missing labelled query set.

---

## 8. Where philosophy returns

Not as asserted history — as **rhetorical framing generated from the grounded
summary**. Once positions and their supporting evidence exist, the system may
pose questions ("these three positions disagree about what constitutes a
measurement — on what grounds?"). Questions are not truth claims and carry no
fabrication risk. Conceptual grounding can then be added by a human, or later by
a sourced retrieval step, never by unsourced generation.

---

## 9. Open decisions (operator)

1. The **class list** and their **weighting hierarchy** (deferred by operator).
2. How to operationalise venue tiers (R1) — derived percentile vs curated list.
3. Whether community discussion is in scope for v1 given R6.
4. Node-count budget and default edge types for v1.

## 10. Build order (walking skeleton first)

1. **L1 harvest** (OpenAlex) — unblocks everything; delivers the bigger corpus.
2. **L4 graph on citations alone** + **L5 render** — smallest end-to-end slice
   that already satisfies the governing principle (traceable pathway) with *one*
   class and no weighting.
3. **L2/L3 signal registry + weighting** — once classes are supplied.
4. **L6 positions + composition chart**.
5. **L7 audit export**, then researcher validation.

Step 2 is deliberately the first *useful* artifact: it produces a real,
inspectable lineage map before any contested weighting exists.
