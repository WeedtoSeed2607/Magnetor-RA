"""Branch A: Topic-Trend Tracking (Spec Section 6).

The spec's engine is gensim ``LdaSeqModel``, which has no Python 3.14 wheel and
won't build without a C++ toolchain (see ADR-0005). This delivers the spec's
*measurable outputs* — topic-probability drift, keyword arrays, cluster data,
and statistic-anchored trend descriptions (Sections 6, 10) — using scikit-learn
instead: fit a shared topic space over the domain's abstracts, then track each
topic's prevalence across time slices.

Faithful to the spec's corrections: refresh is **volume-gated** (not clock-gated),
topic drift is measured (vocabulary/topic-probability), the interpreter is
statistic-anchored with no narrative-mood language, and sentiment is a separate
optional module (deferred, not merged into trend output). Clustering is
within-domain only — nothing crosses domain boundaries.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from magnetor.config import DomainConfig, global_store_path
from magnetor.resources import DomainStore
from magnetor.types import Domain, Paper

TRENDS_FILENAME = "trends.json"
DTM_MODEL_FILENAME = "dtm_model.pkl"
TREND_LOG_FILENAME = "topic_trend_log.json"

DEFAULT_NUM_TOPICS = 5
DEFAULT_SLICE_DAYS = 7
_TOP_KEYWORDS = 8
_MAX_VOCAB = 2000
_MAX_ANOMALIES = 8
#: Minimum documents a time slice needs before its drift/anomaly figures are
#: reported without a low-support caveat. Below this, a slice's mean is a
#: 1-2-document artifact, not a trend — surface that honestly (Spec 6).
_MIN_SLICE_SUPPORT = 3


@dataclass(frozen=True, slots=True)
class TopicTrend:
    """One topic (cluster) and how its prevalence drifts across time slices."""

    topic_id: int
    keywords: tuple[str, ...]
    prevalence: tuple[float, ...]  # mean topic-probability per time slice
    drift: float  # latest slice minus previous (0.0 if <2 slices)


@dataclass(frozen=True, slots=True)
class Anomaly:
    """An emerging term: per-document rate spiked in the latest slice."""

    term: str
    latest_rate: float
    prior_rate: float
    delta: float


@dataclass(frozen=True, slots=True)
class TrendResult:
    domain: Domain
    ran: bool
    reason: str = ""
    n_docs: int = 0
    n_slices: int = 0
    slice_labels: tuple[str, ...] = ()
    topics: tuple[TopicTrend, ...] = ()
    anomalies: tuple[Anomaly, ...] = ()
    interpretation: tuple[str, ...] = ()


def run_trend_analysis(
    config: DomainConfig,
    store: DomainStore,
    *,
    num_topics: int = DEFAULT_NUM_TOPICS,
    slice_days: int = DEFAULT_SLICE_DAYS,
    force: bool = False,
    now: dt.datetime | None = None,
) -> TrendResult:
    """Run one Branch-A trend cycle for a domain (Spec 6 / 12)."""
    moment = now or dt.datetime.now(tz=dt.UTC)
    record_count = store.record_count()
    new_docs = record_count - store.last_trend_count()
    if not force and new_docs < config.trend_min_new_docs:
        return TrendResult(
            domain=config.domain, ran=False,
            reason=f"volume gate: {new_docs} new docs < {config.trend_min_new_docs}",
        )

    dated = [p for p in store.read_records() if p.published is not None and _text(p)]
    if len(dated) < num_topics:
        store.record_trend_run(moment, record_count=record_count)
        return TrendResult(
            domain=config.domain, ran=True, n_docs=len(dated),
            reason=f"too few documents ({len(dated)}) to model {num_topics} topics",
        )

    texts = [_text(p) for p in dated]
    vectorizer = CountVectorizer(
        stop_words="english", min_df=1, max_df=0.9, max_features=_MAX_VOCAB
    )
    try:
        counts = vectorizer.fit_transform(texts)
    except ValueError:
        # e.g. every term pruned by max_df on a near-homogeneous corpus.
        store.record_trend_run(moment, record_count=record_count)
        return TrendResult(
            domain=config.domain, ran=True, n_docs=len(dated),
            reason="no usable vocabulary after stop-word/frequency pruning",
        )
    vocab = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0, max_iter=25)
    doc_topic = lda.fit_transform(counts)

    slice_ids, labels = _slice_ids(dated, slice_days)
    n_slices = len(labels)
    slice_counts = np.bincount(slice_ids, minlength=n_slices)
    topics = _build_topics(lda, vocab, doc_topic, slice_ids, n_slices)
    anomalies = _detect_anomalies(counts.toarray(), vocab, slice_ids, n_slices)
    interpretation = _interpret(topics, anomalies, n_slices, slice_days, slice_counts)

    _persist(
        store, vectorizer, lda, moment, config.domain,
        labels, topics, anomalies, interpretation,
    )
    store.record_trend_run(moment, record_count=record_count)
    return TrendResult(
        domain=config.domain, ran=True, n_docs=len(dated), n_slices=n_slices,
        slice_labels=labels, topics=topics, anomalies=anomalies, interpretation=interpretation,
    )


def _text(paper: Paper) -> str:
    parts = [paper.title.strip(), paper.abstract.strip()]
    return "\n".join(part for part in parts if part)


def _slice_ids(papers: list[Paper], slice_days: int) -> tuple[NDArray[np.int_], tuple[str, ...]]:
    """Assign each paper (in order) to a time slice; return ids + slice labels."""
    published = [p.published for p in papers]
    earliest = min(d for d in published if d is not None)
    step = dt.timedelta(days=slice_days)
    ids = np.array([(d - earliest) // step if d is not None else 0 for d in published], dtype=int)
    n_slices = int(ids.max()) + 1
    labels = tuple(
        (earliest + step * s).date().isoformat() for s in range(n_slices)
    )
    return ids, labels


def _build_topics(
    lda: LatentDirichletAllocation,
    vocab: NDArray[np.str_],
    doc_topic: NDArray[np.float64],
    slice_ids: NDArray[np.int_],
    n_slices: int,
) -> tuple[TopicTrend, ...]:
    num_topics = doc_topic.shape[1]
    prevalence = np.zeros((num_topics, n_slices))
    for s in range(n_slices):
        rows = slice_ids == s
        if rows.any():
            prevalence[:, s] = doc_topic[rows].mean(axis=0)
    topics: list[TopicTrend] = []
    for topic_id in range(num_topics):
        order = np.argsort(lda.components_[topic_id])[::-1][:_TOP_KEYWORDS]
        keywords = tuple(str(vocab[i]) for i in order)
        series = tuple(float(prevalence[topic_id, s]) for s in range(n_slices))
        drift = float(series[-1] - series[-2]) if n_slices >= 2 else 0.0
        topics.append(TopicTrend(topic_id, keywords, series, drift))
    return tuple(topics)


def _detect_anomalies(
    counts: NDArray[np.int_],
    vocab: NDArray[np.str_],
    slice_ids: NDArray[np.int_],
    n_slices: int,
) -> tuple[Anomaly, ...]:
    """Keyword spikes normalised by ingestion volume (Spec 6): per-doc rates."""
    if n_slices < 2:
        return ()
    latest = slice_ids == (n_slices - 1)
    prior = slice_ids < (n_slices - 1)
    n_latest, n_prior = int(latest.sum()), int(prior.sum())
    if n_latest == 0 or n_prior == 0:
        return ()
    rate_latest = counts[latest].sum(axis=0) / n_latest
    rate_prior = counts[prior].sum(axis=0) / n_prior
    delta = rate_latest - rate_prior
    order = np.argsort(delta)[::-1][:_MAX_ANOMALIES]
    return tuple(
        Anomaly(str(vocab[i]), float(rate_latest[i]), float(rate_prior[i]), float(delta[i]))
        for i in order
        if delta[i] > 0
    )


def _interpret(
    topics: tuple[TopicTrend, ...],
    anomalies: tuple[Anomaly, ...],
    n_slices: int,
    slice_days: int,
    slice_counts: NDArray[np.int_],
) -> tuple[str, ...]:
    """Statistic-anchored descriptions with explicit support (Spec 6).

    Every drift/anomaly figure carries the document count behind it, and
    headlines computed from a thin latest slice (< ``_MIN_SLICE_SUPPORT`` docs)
    are marked LOW SUPPORT so a 1-2-document artifact is never dressed up as a
    trend. No narrative-mood language.
    """
    lines: list[str] = []
    latest_n = int(slice_counts[-1]) if len(slice_counts) else 0

    if n_slices < 2:
        lines.append(
            f"Single {slice_days}-day time slice ({latest_n} docs) — topic drift "
            "requires accumulation across multiple windows. Corpus composition:"
        )
        for topic in sorted(topics, key=lambda t: t.prevalence[-1], reverse=True)[:3]:
            keywords = ", ".join(topic.keywords[:4])
            share = topic.prevalence[-1] * 100
            lines.append(f"Topic {topic.topic_id} [{keywords}]: {share:.0f}% of the corpus.")
        return tuple(lines)

    prev_n = int(slice_counts[-2])
    low_support = latest_n < _MIN_SLICE_SUPPORT or prev_n < _MIN_SLICE_SUPPORT
    lines.append(
        f"Latest {slice_days}-day slice holds {latest_n} doc(s); prior slice {prev_n}. "
        + ("LOW SUPPORT — figures below are volatile, interpret with caution."
           if low_support else "Drift below is latest-vs-prior mean topic-probability.")
    )
    movers = sorted(topics, key=lambda t: abs(t.drift), reverse=True)[:3]
    for topic in movers:
        direction = "rose" if topic.drift >= 0 else "fell"
        points = abs(topic.drift) * 100
        keywords = ", ".join(topic.keywords[:4])
        lines.append(
            f"Topic {topic.topic_id} [{keywords}]: prevalence {direction} "
            f"{points:.0f} points (n={latest_n} vs {prev_n})."
        )
    for anomaly in anomalies[:3]:
        caveat = "  [low support]" if latest_n < _MIN_SLICE_SUPPORT else ""
        lines.append(
            f"Emerging term '{anomaly.term}': {anomaly.latest_rate:.2f} mentions/doc "
            f"(latest, n={latest_n}) vs {anomaly.prior_rate:.2f} (prior).{caveat}"
        )
    return tuple(lines)


def _persist(
    store: DomainStore,
    vectorizer: CountVectorizer,
    lda: LatentDirichletAllocation,
    moment: dt.datetime,
    domain: Domain,
    labels: tuple[str, ...],
    topics: tuple[TopicTrend, ...],
    anomalies: tuple[Anomaly, ...],
    interpretation: tuple[str, ...],
) -> None:
    root = store.root
    root.mkdir(parents=True, exist_ok=True)

    trends = {
        "domain": domain.value,
        "generated_at": moment.isoformat(),
        "n_slices": len(labels),
        "slice_labels": list(labels),
        "interpretation": list(interpretation),
        "topics": [
            {
                "id": t.topic_id,
                "keywords": list(t.keywords),
                "prevalence": [round(v, 4) for v in t.prevalence],
                "drift": round(t.drift, 4),
            }
            for t in topics
        ],
        "anomalies": [
            {"term": a.term, "latest_rate": round(a.latest_rate, 3), "delta": round(a.delta, 3)}
            for a in anomalies
        ],
    }
    (root / TRENDS_FILENAME).write_text(json.dumps(trends, indent=2), encoding="utf-8")
    with (root / DTM_MODEL_FILENAME).open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "lda": lda}, handle)
    _append_trend_log(domain, moment, topics, anomalies)


def _append_trend_log(
    domain: Domain,
    moment: dt.datetime,
    topics: tuple[TopicTrend, ...],
    anomalies: tuple[Anomaly, ...],
) -> None:
    """Append a timestamped entry to the global topic_trend_log.json (Spec 10)."""
    path = global_store_path(TREND_LOG_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[object] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            entries = existing
    entries.append(
        {
            "timestamp": moment.isoformat(),
            "domain": domain.value,
            "topics": [
                {"id": t.topic_id, "keywords": list(t.keywords[:4]), "drift": round(t.drift, 4)}
                for t in topics
            ],
            "anomalies": [a.term for a in anomalies[:5]],
        }
    )
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
