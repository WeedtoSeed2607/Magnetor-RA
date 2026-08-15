"""Branch A — what a subject is working on now, grouped by meaning rather than wording.

The trend engine models topics with LDA, which groups words that co-occur. That
answers "which words appear together" and is why its topics read like
``[quantum, based, non, dynamics]``: *based* and *non* are artefacts of counting,
and no amount of tuning turns a word list into a subject a reader recognises.

This groups the **papers** instead. Every embedded paper already has a vector
from retrieval, and two papers arguing about the same thing sit near each other
in that space even when they share no vocabulary. Clustering those vectors gives
groups formed by what the work is *about*; the words come afterwards, only to
name a group that already exists.

**Where it deliberately stops.** This infers a pattern and reports it. It does not
say whether a group is any good, which side of a disagreement is right, or what
the papers argue — that is appraisal, and it belongs to the verdict layer. The
output is "these papers form a group, it is growing, here are its distinctive
words and some members", and nothing beyond.

**Labels name, they do not summarise.** A group is labelled by the words that are
common inside it and rare outside, so generic vocabulary scores zero and drops
out on its own. That yields *"decoherence, superconducting, qubits"* rather than
*"based, non, using"* — a recognisable handle, still not a sentence about what the
group claims. Producing that would need generative summarisation, which this
project has not decided on.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from magnetor.dating import DateQuality, Placement, place, tally
from magnetor.narrowing import STOPWORDS
from magnetor.types import Paper

#: Groups to look for. Enough to separate the strands of one subject, few enough
#: that each still has members worth reading.
DEFAULT_THEMES = 6
#: Days counted as "now". Matches the trend engine's slice so the two agree about
#: what recent means.
DEFAULT_WINDOW_DAYS = 30
#: Papers needed per group before clustering is worth attempting at all.
MIN_PER_THEME = 3
DEFAULT_LABEL_TERMS = 4
DEFAULT_EXAMPLES = 3

_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")

#: LaTeX control sequences. 107 of the 394 stored abstracts carry them, and
#: ``\mathbb{R}`` tokenises to the word *mathbb*, which then labelled the largest
#: maths group. Stripping the command form is precise: it removes the markup and
#: leaves an ordinary use of the same letters (``log-normal``) untouched.
_LATEX = re.compile(r"\\[a-zA-Z]+")

#: Closed-class English — determiners, pronouns, conjunctions, auxiliaries,
#: quantifiers. The rule for adding here is grammatical, not taste: a word that
#: cannot head a noun phrase cannot name a subject, in any domain.
#:
#: These need their own list because the outside-rate subtraction in
#: :func:`_label_for` cannot reach them. That mechanism removes words spread
#: evenly across groups; *there*, *but* and *every* cluster with theorem-writing
#: style, so they read as genuinely distinctive of a group — of its prose, not of
#: its subject. Measured before adding: they took 3 of maths' 24 label slots.
#:
#: sklearn's ``ENGLISH_STOP_WORDS`` was the obvious shortcut and is not used: it
#: carries *system*, *computer*, *interest* and *fire*, each of which names a real
#: subject in this corpus.
_FUNCTION_WORDS = frozenset(
    {
        "every", "each", "all", "any", "some", "none", "another", "other",
        "others", "either", "neither", "enough", "one", "ones", "two", "three",
        "but", "yet", "nor", "because", "since", "unless", "though", "whereas",
        "there", "here", "then", "where", "again", "further", "once", "still",
        "already", "always", "never", "often", "rather", "quite", "very", "too",
        "only", "even", "just", "more", "most", "less", "least",
        "will", "shall", "would", "could", "should", "must", "might",
        "have", "has", "had", "been", "being", "did", "done",
        "they", "them", "him", "her", "his", "who", "whom", "whose", "its",
    }
)

#: Academic boilerplate on top of the shared stopwords. These survive frequency
#: scoring because they are spread evenly across every group, so subtracting the
#: outside rate leaves them looking distinctive when one group dominates the
#: corpus. They name no subject and must never label one.
_BOILERPLATE = frozenset(
    {
        "abstract", "here", "however", "thus", "therefore", "hence", "moreover",
        "furthermore", "across", "within", "also", "well", "work", "works",
        "present", "presents", "presented", "propose", "proposed", "proposes",
        "show", "shows", "shown", "demonstrate", "demonstrates", "found",
        "finding", "findings", "obtained", "observed", "reported", "given",
        "develop", "developed", "developing", "provide", "provides", "provided",
        "become", "becomes", "promising", "important", "significant", "recent",
        "recently", "respectively", "compared", "comparison", "applicability",
        "building", "contributions", "implications", "considered", "various",
        "different", "several", "many", "much", "such", "both", "including",
        "than", "while", "whether", "although", "significantly", "associated",
        "involved", "related", "towards", "toward",
    }
)


@dataclass(frozen=True, slots=True)
class Theme:
    """One group of papers that mean similar things, and how it is moving."""

    label: tuple[str, ...]  # words common inside, rare outside
    size: int
    #: Members placed inside the window. ``recent + earlier`` can be less than
    #: ``size``: a member whose date is missing, too coarse or in the future
    #: still belongs to the group but cannot sit on either side of the cutoff.
    recent: int
    earlier: int
    share_now: float  # of all recent papers, the fraction in this group
    share_before: float
    examples: tuple[str, ...]  # external ids nearest the group's centre
    #: Every member, nearest the centre first. ``examples`` is the head of this.
    #: Needed whole by :mod:`magnetor.naming`: a reader asked to name a group has
    #: to be shown the group, and a sample would have them name the sample.
    members: tuple[str, ...] = ()

    @property
    def drift(self) -> float:
        """Change in share of the subject's output. Positive means gaining ground."""
        return self.share_now - self.share_before

    @property
    def name(self) -> str:
        return ", ".join(self.label) if self.label else "(no distinctive words)"


@dataclass(frozen=True, slots=True)
class ThemeReport:
    themes: tuple[Theme, ...]
    papers: int
    window_days: int
    reason: str = ""  # why there is nothing to show, when there is nothing
    #: What the before-and-after comparison could actually use.
    dates: DateQuality = field(default_factory=DateQuality)
    #: Days between the oldest and newest placed paper. A corpus spanning less
    #: than ``window_days`` cannot support a before-and-after claim at all — the
    #: cutoff falls inside a single batch and splits it arbitrarily.
    span_days: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.themes)

    @property
    def spans_the_window(self) -> bool:
        """Whether the corpus is long enough for ``drift`` to mean anything."""
        return self.span_days >= self.window_days


def _terms(text: str) -> set[str]:
    ignored = STOPWORDS | _BOILERPLATE | _FUNCTION_WORDS
    stripped = _LATEX.sub(" ", text).lower()
    return {w for w in _WORD.findall(stripped) if w not in ignored}


def _label_for(
    members: Sequence[int],
    documents: Sequence[set[str]],
    outside_frequency: Mapping[str, int],
    outside_total: int,
    *,
    terms: int,
) -> tuple[str, ...]:
    """Words common inside the group and rare outside it.

    A plain "most frequent words" label returns the subject's own vocabulary for
    every group — every quantum paper says *quantum*. Subtracting the outside
    rate is what makes a label distinguish one group from its neighbours.
    """
    inside: Counter[str] = Counter()
    for index in members:
        inside.update(documents[index])
    n_in = len(members)
    if not n_in:
        return ()
    scored = [
        (
            word / n_in - outside_frequency.get(term, 0) / max(1, outside_total),
            term,
        )
        for term, word in ((t, c) for t, c in inside.items() if c >= 2)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(term for score, term in scored[:terms] if score > 0)


def discover_themes(
    ids: Sequence[str],
    vectors: NDArray[np.float32],
    papers: Mapping[str, Paper],
    *,
    n_themes: int = DEFAULT_THEMES,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: dt.datetime | None = None,
    label_terms: int = DEFAULT_LABEL_TERMS,
    examples: int = DEFAULT_EXAMPLES,
) -> ThemeReport:
    """Group embedded papers by meaning and report which groups are growing.

    Pure: the caller supplies ids, their vectors and the matching records, so the
    grouping can be exercised without a corpus on disk.
    """
    usable = [(i, pid) for i, pid in enumerate(ids) if pid in papers]
    if len(usable) < n_themes * MIN_PER_THEME:
        return ThemeReport(
            (),
            len(usable),
            window_days,
            reason=(
                f"{len(usable)} embedded paper(s) is too few to separate "
                f"{n_themes} groups; grouping needs about "
                f"{n_themes * MIN_PER_THEME}."
            ),
        )

    rows = np.asarray([vectors[i] for i, _pid in usable], dtype=np.float64)
    members_ids = [pid for _i, pid in usable]
    # Ward linkage, measured on unit-length vectors so Euclidean distance stands in
    # for cosine. Average linkage was tried first and chained badly: papers within
    # one subject are all fairly similar, so it produced one cluster holding 108 of
    # 130 papers plus a scatter of singletons — which is not a set of themes.
    # Ward minimises within-cluster variance and so resists that. It stays
    # deterministic, which K-means would not: the same corpus must give the same
    # groups or no statement about growth is reproducible.
    labels = AgglomerativeClustering(
        n_clusters=min(n_themes, len(rows)), linkage="ward"
    ).fit_predict(rows)

    documents = [
        _terms(f"{papers[pid].title} {papers[pid].abstract or ''}") for pid in members_ids
    ]
    corpus_frequency: Counter[str] = Counter()
    for doc in documents:
        corpus_frequency.update(doc)

    moment = now or dt.datetime.now(tz=dt.UTC)
    # Placement is decided once per paper, not recomputed per group, so a paper
    # counted as recent for the corpus is the same paper counted as recent for
    # its theme. The shares would not sum to 1 otherwise.
    placements = {
        pid: place(
            papers[pid].published,
            papers[pid].published_precision,
            now=moment,
            window_days=window_days,
        )
        for pid in members_ids
    }
    placed = {Placement.RECENT, Placement.EARLIER}
    unrecorded = sum(
        1
        for pid in members_ids
        if papers[pid].published_precision is None and placements[pid] in placed
    )
    quality = tally(placements, unrecorded)
    total_recent = quality.recent
    total_earlier = quality.earlier

    placed_dates = [
        papers[pid].published for pid in members_ids if placements[pid] in placed
    ]
    known = [stamp for stamp in placed_dates if stamp is not None]
    span = (max(known) - min(known)).days if known else 0

    found: list[Theme] = []
    for cluster in sorted(set(labels.tolist())):
        indexes = [i for i, value in enumerate(labels) if value == cluster]
        group_ids = [members_ids[i] for i in indexes]
        recent = sum(1 for pid in group_ids if placements[pid] is Placement.RECENT)
        earlier = sum(1 for pid in group_ids if placements[pid] is Placement.EARLIER)

        inside_words: Counter[str] = Counter()
        for i in indexes:
            inside_words.update(documents[i])
        outside_frequency = corpus_frequency - inside_words

        centroid = rows[indexes].mean(axis=0)
        order = np.argsort(-(rows[indexes] @ centroid))
        by_closeness = tuple(group_ids[int(i)] for i in order)
        found.append(
            Theme(
                label=_label_for(
                    indexes,
                    documents,
                    outside_frequency,
                    len(members_ids) - len(indexes),
                    terms=label_terms,
                ),
                size=len(group_ids),
                recent=recent,
                earlier=earlier,
                share_now=recent / total_recent if total_recent else 0.0,
                share_before=earlier / total_earlier if total_earlier else 0.0,
                examples=by_closeness[:examples],
                members=by_closeness,
            )
        )

    # Growing first: the question is what is happening now, not what is largest.
    # Unless the corpus is shorter than the window, in which case "growing" is not
    # a fact about the subject at all — the cutoff falls inside a single batch and
    # splits it arbitrarily. Order by size then, which is at least true.
    if span >= window_days:
        found.sort(key=lambda t: (-t.drift, -t.recent))
    else:
        found.sort(key=lambda t: (-t.size, t.name))
    return ThemeReport(
        tuple(found), len(members_ids), window_days, dates=quality, span_days=span
    )
