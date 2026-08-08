"""Branch C — how a paper approaches its problem, not what it is about.

The operator's concern 1: researchers converge on one niche and cannot borrow
from adjacent ones. A citation graph cannot fix that by itself, because it only
knows *who cited whom* — two papers attacking the same question by empirical and
by formal means look identical to it. This module adds the missing axis: the
**mode of approach**.

**Why not Aristotle's four causes as proposed.** They are four *answerabilities*
(``aitia`` is "that which is answerable for"), not four categories of thing, and
Aristotle himself says form, end and mover "often coincide" (*Physics* 198a24) —
so they are not orthogonal and cannot serve as independent axes. They are also
near-isomorphic to Levenstein's descriptive/mechanistic/normative trichotomy,
which the operator's own critique document refutes for exhaustiveness: a
renormalisation-group explanation is none of the three. Two consequences are
designed in rather than argued away: facets are **multi-label**, because a paper
can be empirical *and* formal without contradiction, and ``unclassified`` is a
first-class outcome, following the coding manual's ``none-of-the-above``.

**Where the invariant sits.** Classification happens during the harvest, while
abstracts are still in memory, and only the resulting label plus the terms that
triggered it are persisted. A graph artifact therefore still carries identifiers,
metrics and edges and never a document body (ADR-0006 section 3 / I4). Storing
the matched terms is what makes an assignment checkable rather than asserted —
C1, no output without its inputs.

**Honest status: this is a screening heuristic, not a validated instrument.**
Lexicon matching is transparent and free, and it needs no LLM gateway (still
undecided). It has not been measured against human coders, so the manual's
reliability floor is unmet and no conclusion should rest on a facet alone. The
evidence terms are stored precisely so a reader can overrule it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from magnetor.harvest import HarvestResult

EMPIRICAL = "empirical"
FORMAL = "formal"
MECHANISTIC = "mechanistic"
NORMATIVE = "normative"
CONCEPTUAL = "conceptual"
LINGUISTIC = "linguistic"
UNCLASSIFIED = "unclassified"

#: Distinct terms a facet needs before it is assigned. One match is a coincidence;
#: the bar is deliberately low because facets screen rather than adjudicate.
DEFAULT_MIN_TERMS = 2

FACET_LABELS = {
    EMPIRICAL: "Empirical — measures, observes, tests",
    FORMAL: "Formal — proves, derives, models mathematically",
    MECHANISTIC: "Mechanistic — describes how the system works",
    NORMATIVE: "Normative — asks what it is for, or what is optimal",
    CONCEPTUAL: "Conceptual — defines, distinguishes, argues",
    LINGUISTIC: "Linguistic — language, discourse, corpora",
    UNCLASSIFIED: "Unclassified — no facet reached the evidence bar",
}

# Deliberately domain-general: this corpus spans quantum computing, moral
# psychology, empathy and machine learning, and a lexicon tuned to one of them
# would silently misread the others.
_LEXICON: dict[str, frozenset[str]] = {
    EMPIRICAL: frozenset({
        "experiment", "experimental", "experiments", "empirical", "measured",
        "measurement", "measurements", "observed", "observation", "observations",
        "trial", "trials", "participants", "subjects", "sample", "samples",
        "dataset", "datasets", "survey", "fmri", "imaging", "recorded",
        "benchmark", "evaluation", "replication",
    }),
    FORMAL: frozenset({
        "theorem", "proof", "proved", "lemma", "formal", "formalism", "axiom",
        "equation", "equations", "algebraic", "geometry", "topological",
        "calculus", "probability", "bound", "bounds", "complexity", "algorithm",
        "optimization", "convergence", "asymptotic", "stochastic", "matrix",
    }),
    MECHANISTIC: frozenset({
        "mechanism", "mechanisms", "circuit", "circuits", "pathway", "pathways",
        "substrate", "neural", "molecular", "dynamics", "implementation",
        "architecture", "underlying", "cortex", "synaptic", "computation",
    }),
    NORMATIVE: frozenset({
        "optimal", "optimality", "utility", "adaptive", "purpose", "selection",
        "evolutionary", "rational", "rationality", "norm", "norms", "normative",
        "ought", "moral", "ethical", "value", "values", "goal", "goals", "fitness",
    }),
    CONCEPTUAL: frozenset({
        "concept", "concepts", "conceptual", "definition", "defining", "define",
        "philosophical", "philosophy", "ontology", "epistemic", "distinction",
        "taxonomy", "framework", "argument", "critique", "notion",
    }),
    LINGUISTIC: frozenset({
        "language", "linguistic", "linguistics", "semantic", "semantics",
        "discourse", "corpus", "lexical", "narrative", "syntax", "pragmatics",
        "translation", "wording",
    }),
}

_WORD = re.compile(r"[a-z][a-z-]+")


@dataclass(frozen=True, slots=True)
class FacetAssignment:
    """One paper's facets, with the terms that produced each — auditable by design."""

    openalex_id: str
    facets: tuple[str, ...]
    evidence: dict[str, tuple[str, ...]]
    #: Whether an abstract was available to read. Measured on this corpus: papers
    #: without one are 96.5% unclassified against 43.2% for those with one, so the
    #: unclassified rate is uninterpretable without it. A boolean about
    #: availability, not the text itself (C3 — absence is displayed, not imputed).
    had_abstract: bool = False


@dataclass(frozen=True, slots=True)
class FacetIndex:
    assignments: tuple[FacetAssignment, ...]

    def by_id(self) -> dict[str, FacetAssignment]:
        return {a.openalex_id: a for a in self.assignments}


def classify_text(
    title: str, abstract: str = "", *, min_terms: int = DEFAULT_MIN_TERMS
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Facets for one paper, plus the matched terms per facet.

    The abstract carries most of the signal; a title alone rarely names its own
    method. When no abstract is available the result is correspondingly weaker,
    which is why ``unclassified`` must stay a legitimate answer.
    """
    words = set(_WORD.findall(f"{title} {abstract}".lower()))
    evidence: dict[str, tuple[str, ...]] = {}
    for facet, terms in _LEXICON.items():
        hits = tuple(sorted(words & terms))
        if len(hits) >= min_terms:
            evidence[facet] = hits
    if not evidence:
        return (UNCLASSIFIED,), {}
    # Strongest evidence first, so a reader sees the best-supported facet first.
    ordered = sorted(evidence, key=lambda f: (-len(evidence[f]), f))
    return tuple(ordered), evidence


def classify(
    result: HarvestResult, *, min_terms: int = DEFAULT_MIN_TERMS
) -> FacetIndex:
    """Classify every harvested paper while abstracts are still in memory."""
    assignments = []
    for paper in result.papers:
        facets, evidence = classify_text(
            paper.title, paper.abstract, min_terms=min_terms
        )
        assignments.append(
            FacetAssignment(
                openalex_id=paper.openalex_id, facets=facets, evidence=evidence,
                had_abstract=bool(paper.abstract.strip()),
            )
        )
    return FacetIndex(tuple(assignments))


def facet_counts(nodes: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """How many papers carry each facet. Multi-label, so this oversums the node count."""
    counts: dict[str, int] = {}
    for node in nodes:
        raw = node.get("facets")
        for facet in raw if isinstance(raw, list) else [UNCLASSIFIED]:
            counts[str(facet)] = counts.get(str(facet), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def unclassified_breakdown(
    nodes: Sequence[Mapping[str, object]],
) -> tuple[int, int]:
    """``(unclassified, of those with no abstract to read)``.

    Without the second number the first looks like a broken classifier rather
    than what it mostly is: OpenAlex holding no abstract for the work.
    """
    total = blind = 0
    for node in nodes:
        if node_facets(node) == (UNCLASSIFIED,):
            total += 1
            if not node.get("had_abstract"):
                blind += 1
    return total, blind


def node_facets(node: Mapping[str, object]) -> tuple[str, ...]:
    raw = node.get("facets")
    if isinstance(raw, list) and raw:
        return tuple(str(f) for f in raw)
    return (UNCLASSIFIED,)


def cross_facet_neighbours(
    node_id: str,
    related: Sequence[tuple[str, int]],
    nodes: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    """Related papers that approach the shared subject *differently*.

    This is the concern-1 payoff: a paper is coupled to another (they lean on the
    same earlier work) but classified under a facet the first one does not carry.
    Same problem, different mode of attack — the borrowing a keyword search cannot
    surface, because the vocabularies differ precisely where the approaches do.
    """
    own = set(node_facets(nodes.get(node_id, {})))
    out: list[tuple[str, int, tuple[str, ...]]] = []
    for other_id, weight in related:
        other = node_facets(nodes.get(other_id, {}))
        different = tuple(f for f in other if f not in own and f != UNCLASSIFIED)
        if different:
            out.append((other_id, weight, different))
    return tuple(out)
