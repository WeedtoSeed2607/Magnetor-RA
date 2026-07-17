"""Branch A Topic-Trend Tracking (Spec 6): volume gate, drift, anomalies, persist."""

from __future__ import annotations

import datetime as dt
import json

from magnetor.config import get_domain_config, global_store_path
from magnetor.resources import DomainStore
from magnetor.trends import TREND_LOG_FILENAME, run_trend_analysis
from magnetor.types import Domain, Paper
from tests.conftest import make_paper

_QM = Domain.QUANTUM_MECHANICS


def _store() -> DomainStore:
    return DomainStore(_QM, get_domain_config(_QM).storage_dir)


def _paper(ext: str, text: str, day: int) -> Paper:
    published = dt.datetime(2026, 6, day, tzinfo=dt.UTC)
    paper = make_paper(external_id=ext, abstract=text, published=published)
    object.__setattr__(paper, "title", "")  # keep the vocabulary to the abstract
    return paper


def _seed_two_slices(store: DomainStore) -> None:
    # Slice 0 = days 1-6 (bin 0); slice 1 = days 8-13 (bin 1), 7-day slices.
    for i in range(6):
        store.store(
            _paper(f"w1-{i}", "quantum error correction surface code decoding stabilizer", 1 + i)
        )
    for i in range(6):
        store.store(
            _paper(f"w2-{i}", "graph coloring chromatic transformer neural network", 8 + i)
        )


def test_volume_gate_blocks_below_threshold() -> None:
    store = _store()
    for i in range(5):  # 5 new docs < default min 10
        store.store(_paper(f"p{i}", "quantum error correction", 1 + i))
    result = run_trend_analysis(get_domain_config(_QM), store)
    assert result.ran is False
    assert "volume gate" in result.reason


def test_force_runs_and_measures_drift() -> None:
    store = _store()
    _seed_two_slices(store)
    result = run_trend_analysis(get_domain_config(_QM), store, num_topics=2, force=True)

    assert result.ran is True
    assert result.n_slices == 2
    assert len(result.topics) == 2
    for topic in result.topics:
        assert len(topic.prevalence) == 2  # one value per slice
    # At least one topic drifted between the two slices.
    assert any(abs(t.drift) > 0.01 for t in result.topics)
    assert result.interpretation  # statistic-anchored lines produced


def test_anomaly_flags_emerging_term() -> None:
    store = _store()
    _seed_two_slices(store)
    result = run_trend_analysis(get_domain_config(_QM), store, num_topics=2, force=True)
    terms = {a.term for a in result.anomalies}
    # "transformer" appears only in the latest slice -> should surface as emerging.
    assert "transformer" in terms


def test_too_few_documents_runs_but_no_topics() -> None:
    store = _store()
    for i in range(3):
        store.store(_paper(f"p{i}", "quantum error correction", 1 + i))
    result = run_trend_analysis(get_domain_config(_QM), store, num_topics=5, force=True)
    assert result.ran is True
    assert result.topics == ()
    assert "too few documents" in result.reason


def test_persistence_writes_trends_model_and_global_log(tmp_path) -> None:
    store = _store()
    _seed_two_slices(store)
    run_trend_analysis(get_domain_config(_QM), store, num_topics=2, force=True)

    root = store.root
    assert (root / "trends.json").exists()
    assert (root / "dtm_model.pkl").exists()
    log = global_store_path(TREND_LOG_FILENAME)
    assert log.exists()
    entries = json.loads(log.read_text(encoding="utf-8"))
    assert isinstance(entries, list) and entries[-1]["domain"] == "qm"


def test_trend_state_advances() -> None:
    store = _store()
    _seed_two_slices(store)  # 12 docs
    assert store.last_trend_count() == 0
    run_trend_analysis(get_domain_config(_QM), store, num_topics=2, force=True)
    assert store.last_trend_count() == 12
    # A second run with no new docs is volume-gated.
    second = run_trend_analysis(get_domain_config(_QM), store, num_topics=2)
    assert second.ran is False


def test_single_slice_notes_insufficient_span() -> None:
    store = _store()
    # All same day -> one slice; distinct vocab so it survives max_df pruning.
    topics = [
        "quantum error correction surface codes",
        "graph coloring chromatic number",
        "protein folding molecular dynamics",
        "neural network transformer attention",
        "black hole event horizon thermodynamics",
        "prime numbers riemann zeta function",
    ]
    for i, text in enumerate(topics):
        store.store(_paper(f"p{i}", text, 1))
    result = run_trend_analysis(get_domain_config(_QM), store, num_topics=2, force=True)
    assert result.ran is True
    assert result.n_slices == 1
    assert any("Single" in line for line in result.interpretation)
    # Single-slice banner is now informative: shows corpus composition.
    assert any("% of the corpus" in line for line in result.interpretation)


def test_interpretation_annotates_support_and_flags_low_support() -> None:
    store = _store()
    # Slice 0 = days 1-6 (6 docs, high support); slice 1 = day 8 (1 doc, thin).
    for i in range(6):
        store.store(
            _paper(f"a{i}", "quantum error correction surface code stabilizer decoding", 1 + i)
        )
    store.store(_paper("b0", "graph coloring chromatic transformer neural network", 8))
    result = run_trend_analysis(get_domain_config(_QM), store, num_topics=2, force=True)

    assert result.n_slices == 2
    text = "\n".join(result.interpretation)
    # Latest slice has 1 doc (< _MIN_SLICE_SUPPORT) -> must be flagged, not silent.
    assert "LOW SUPPORT" in text
    # Every drift figure carries its document count.
    assert "n=1 vs 6" in text
