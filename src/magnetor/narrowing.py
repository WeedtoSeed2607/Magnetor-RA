"""Branch C — is this query too broad, and what would narrow it?

The operator's concern 2: *"If the keyword is too broad, give recommendations as
to narrow it down to what specific question."*

**Breadth is measured from the graph, not guessed from the query string.** A
keyword is too broad when the papers it gathered do not cite one another: the
harvest returned several literatures side by side rather than one. That is
directly observable — pair coverage, and the share of papers sitting outside the
main component — and it is the signal a plain keyword search cannot produce,
because it requires the citation structure. Fragmentation is deliberately a
*share* rather than a component count: a sixty-node graph with one detached
straggler has two components and is plainly still one literature.

**Suggestions are ranked by cohesion, not frequency.** The common move is to
offer the most frequent phrases, which mostly returns the query back with
decoration. Here a candidate phrase is scored by whether *the papers containing
it cite each other more densely than the graph as a whole*. A phrase that picks
out a genuine sub-literature scores above 1; a phrase that merely occurs often
scores around 1 and is discarded. Narrowing to a real sub-literature is the point;
narrowing to a common word is not.

**The thresholds here are declared conventions, not findings.** ADR-0006 L8 is
explicit that inventing cutoffs and presenting them as established is the false
precision the architecture exists to avoid. Every bound below is a named default
and a parameter, and the verdict is reported alongside the raw numbers so the
reader can disagree with the cutoff and still use the measurement.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from magnetor.pathways import GraphView, graph_view

FOCUSED = "focused"
BROAD = "broad"
SCATTERED = "scattered"

#: Declared conventions (see module docstring), all overridable.
DEFAULT_BROAD_COVERAGE = 0.08  # pair coverage below this reads as more than one literature
DEFAULT_SCATTERED_COVERAGE = 0.04
#: A phrase must appear in at least this many papers to be a candidate sub-topic.
DEFAULT_MIN_PAPERS = 3
#: ...and in no more than this share, or it restates the query instead of narrowing it.
DEFAULT_MAX_SHARE = 0.6
#: Cohesion at or below this means the phrase groups no better than chance.
DEFAULT_MIN_COHESION = 1.2
DEFAULT_MAX_PHRASE_WORDS = 3
#: Pulls small groups toward the graph-wide density when ranking. See
#: :attr:`Suggestion.score` for why a raw ratio cannot be ranked directly.
_SHRINKAGE = 8.0
#: Share of papers that may sit outside the largest component before the graph
#: reads as fragmented. A couple of stray nodes is normal; a third of the set
#: sitting apart means the query gathered separate literatures.
DEFAULT_BROAD_FRAGMENTATION = 0.05
DEFAULT_SCATTERED_FRAGMENTATION = 0.15

_WORD = re.compile(r"[a-z][a-z0-9-]+")

# Broader than the pathway seed's list: phrase extraction needs to drop the
# connective tissue that would otherwise head every n-gram.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "by",
        "from", "at", "as", "is", "are", "be", "was", "were", "its", "their", "this",
        "that", "these", "those", "we", "our", "using", "used", "use", "via", "into",
        "over", "under", "between", "about", "new", "novel", "toward", "towards",
        "study", "studies", "review", "analysis", "approach", "approaches", "case",
        "evidence", "effect", "effects", "role", "based", "results", "data", "model",
        "models", "method", "methods", "research", "paper", "article", "report",
        "can", "may", "not", "how", "why", "what", "when", "which", "does", "do",
    }
)


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A candidate narrowing, with the evidence for it."""

    phrase: str
    papers: int
    share: float  # fraction of the graph carrying the phrase
    cohesion: float  # internal link density / graph-wide density
    examples: tuple[str, ...]

    @property
    def score(self) -> float:
        """Cohesion, damped for small samples and biased toward usable phrases.

        Raw cohesion is unusable as a ranking on its own: three papers have three
        possible pairs, so one citation between them yields a spectacular ratio.
        The shrinkage factor ``n/(n+k)`` pulls small groups toward the graph mean,
        which is what keeps an incidental word from a single title off the top of
        the list. The length bonus prefers multi-word phrases because the output
        is meant to be pasted back as a narrower query, and one common word is not
        a narrower question.
        """
        shrunk = self.cohesion * (self.papers / (self.papers + _SHRINKAGE))
        words = len(self.phrase.split())
        return shrunk * (1.0 + 0.5 * (words - 1))


@dataclass(frozen=True, slots=True)
class BreadthReport:
    verdict: str  # FOCUSED | BROAD | SCATTERED
    nodes: int
    linked_pairs: int
    pair_coverage: float
    components: int
    largest_component_share: float
    suggestions: tuple[Suggestion, ...]

    @property
    def is_broad(self) -> bool:
        return self.verdict != FOCUSED


def _undirected(view: GraphView) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for citing, cited in view.edges:
        adjacency[citing].add(cited)
        adjacency[cited].add(citing)
    return adjacency


def _components(view: GraphView) -> list[set[str]]:
    adjacency = _undirected(view)
    seen: set[str] = set()
    groups: list[set[str]] = []
    for start in view.ids():
        if start in seen:
            continue
        group: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            group.add(node)
            stack.extend(adjacency.get(node, set()) - seen)
        groups.append(group)
    return groups


def _tokens(title: str) -> list[str]:
    return [
        word
        for word in _WORD.findall(title.lower())
        if len(word) > 2 and word not in _STOPWORDS
    ]


def _phrases(title: str, *, max_words: int) -> set[str]:
    """Content-word n-grams. Stopwords are dropped first, so "theory of mind"
    yields "theory mind" — standard for phrase extraction and still readable."""
    words = _tokens(title)
    found: set[str] = set()
    for size in range(1, max_words + 1):
        for start in range(len(words) - size + 1):
            found.add(" ".join(words[start : start + size]))
    return found


def _density(pairs: int, members: int) -> float:
    possible = members * (members - 1) // 2
    return pairs / possible if possible else 0.0


def assess(
    document: Mapping[str, object],
    *,
    broad_coverage: float = DEFAULT_BROAD_COVERAGE,
    scattered_coverage: float = DEFAULT_SCATTERED_COVERAGE,
    broad_fragmentation: float = DEFAULT_BROAD_FRAGMENTATION,
    scattered_fragmentation: float = DEFAULT_SCATTERED_FRAGMENTATION,
    min_papers: int = DEFAULT_MIN_PAPERS,
    max_share: float = DEFAULT_MAX_SHARE,
    min_cohesion: float = DEFAULT_MIN_COHESION,
    max_phrase_words: int = DEFAULT_MAX_PHRASE_WORDS,
    limit: int = 6,
) -> BreadthReport:
    """Judge whether a graph's query was too broad, and propose narrowings."""
    view = graph_view(document)
    nodes = len(view.nodes)
    linked = {frozenset((u, v)) for u, v in view.edges}
    coverage = _density(len(linked), nodes)
    groups = _components(view)
    largest = max((len(g) for g in groups), default=0)
    largest_share = largest / nodes if nodes else 0.0

    # Fragmentation is measured as the share of papers *outside* the main
    # component, not as a component count. A 60-node graph with one detached
    # straggler has four components and is not thereby two literatures.
    fragmentation = 1.0 - largest_share
    if nodes < 2:
        verdict = FOCUSED
    elif coverage < scattered_coverage or fragmentation > scattered_fragmentation:
        verdict = SCATTERED
    elif coverage < broad_coverage or fragmentation > broad_fragmentation:
        verdict = BROAD
    else:
        verdict = FOCUSED

    return BreadthReport(
        verdict=verdict,
        nodes=nodes,
        linked_pairs=len(linked),
        pair_coverage=coverage,
        components=len(groups),
        largest_component_share=largest_share,
        suggestions=_suggest(
            view, linked, coverage,
            min_papers=min_papers, max_share=max_share,
            min_cohesion=min_cohesion, max_phrase_words=max_phrase_words, limit=limit,
        ),
    )


def _suggest(
    view: GraphView,
    linked: set[frozenset[str]],
    graph_coverage: float,
    *,
    min_papers: int,
    max_share: float,
    min_cohesion: float,
    max_phrase_words: int,
    limit: int,
) -> tuple[Suggestion, ...]:
    nodes = len(view.nodes)
    if nodes < min_papers or graph_coverage <= 0:
        return ()

    holders: dict[str, set[str]] = defaultdict(set)
    titles = {node.id: node.title for node in view.nodes}
    for node in view.nodes:
        for phrase in _phrases(node.title, max_words=max_phrase_words):
            holders[phrase].add(node.id)

    scored: list[Suggestion] = []
    for phrase, members in holders.items():
        count = len(members)
        if count < min_papers or count / nodes > max_share:
            continue
        internal = sum(
            1 for a, b in combinations(sorted(members), 2) if frozenset((a, b)) in linked
        )
        cohesion = _density(internal, count) / graph_coverage
        if cohesion < min_cohesion:
            continue
        examples = tuple(
            titles[nid] for nid in sorted(members, key=lambda n: titles[n])[:3]
        )
        scored.append(
            Suggestion(
                phrase=phrase, papers=count, share=count / nodes,
                cohesion=round(cohesion, 2), examples=examples,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return tuple(_drop_nested(scored)[:limit])


def _drop_nested(suggestions: Sequence[Suggestion]) -> list[Suggestion]:
    """Keep the most specific phrasing when one contains another.

    "surface code" and "surface" typically cover nearly the same papers; offering
    both spends a suggestion slot on no extra information.
    """
    kept: list[Suggestion] = []
    for candidate in suggestions:
        words = set(candidate.phrase.split())
        if any(
            words <= set(other.phrase.split()) or set(other.phrase.split()) <= words
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept
