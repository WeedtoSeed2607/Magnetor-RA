"""Pipeline: cadence gate, dedup accounting, and state recording."""

from __future__ import annotations

import datetime as dt

from magnetor.config import get_domain_config
from magnetor.pipeline import run_acquisition
from magnetor.resources import DomainStore
from magnetor.types import Domain
from tests.conftest import FakeSource, make_paper


def _store(tmp_path, domain=Domain.QUANTUM_MECHANICS) -> DomainStore:
    return DomainStore(domain, tmp_path / "data" / domain.value)


def test_first_run_stores_all(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    papers = [make_paper(external_id=f"2601.{i:05d}") for i in range(3)]
    source = FakeSource(Domain.QUANTUM_MECHANICS, papers)
    store = _store(tmp_path)

    result = run_acquisition(config, source, store)

    assert result.ran is True
    assert result.fetched == 3
    assert result.stored == 3
    assert result.skipped_duplicates == 0
    assert store.last_run() is not None


def test_cadence_gate_blocks_early_rerun(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    source = FakeSource(Domain.QUANTUM_MECHANICS, [make_paper()])
    store = _store(tmp_path)

    start = dt.datetime(2026, 6, 20, 12, tzinfo=dt.UTC)
    first = run_acquisition(config, source, store, now=start)
    assert first.ran is True

    # Only an hour later — well inside the 24h cadence.
    second = run_acquisition(
        config, source, store, now=start + dt.timedelta(hours=1)
    )
    assert second.ran is False
    assert "cadence gate" in second.reason


def test_force_bypasses_cadence_gate(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    source = FakeSource(Domain.QUANTUM_MECHANICS, [make_paper()])
    store = _store(tmp_path)

    start = dt.datetime(2026, 6, 20, 12, tzinfo=dt.UTC)
    run_acquisition(config, source, store, now=start)
    forced = run_acquisition(
        config, source, store, now=start + dt.timedelta(hours=1), force=True
    )
    assert forced.ran is True


def test_duplicates_not_double_stored(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    paper = make_paper(external_id="2601.00042")
    store = _store(tmp_path)

    start = dt.datetime(2026, 6, 20, 12, tzinfo=dt.UTC)
    run_acquisition(config, FakeSource(Domain.QUANTUM_MECHANICS, [paper]), store, now=start)

    # A day later the same paper comes back; it must be counted, not re-stored.
    later = start + dt.timedelta(days=1, hours=1)
    result = run_acquisition(
        config, FakeSource(Domain.QUANTUM_MECHANICS, [paper]), store, now=later
    )
    assert result.ran is True
    assert result.fetched == 1
    assert result.stored == 0
    assert result.skipped_duplicates == 1


def test_since_uses_watermark_not_run_time(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    store = _store(tmp_path)
    start = dt.datetime(2026, 6, 20, 12, tzinfo=dt.UTC)
    published = dt.datetime(2026, 6, 18, tzinfo=dt.UTC)

    src1 = FakeSource(Domain.QUANTUM_MECHANICS, [make_paper(published=published)])
    run_acquisition(config, src1, store, now=start)
    assert src1.calls[0][0] is None  # cold start (no watermark yet)

    src2 = FakeSource(Domain.QUANTUM_MECHANICS, [make_paper(external_id="2601.99999")])
    run_acquisition(config, src2, store, now=start + dt.timedelta(days=2), force=True)
    # `since` is the newest paper actually seen, not the wall-clock run time.
    assert src2.calls[0][0] == published


def test_watermark_advances_to_newest_published(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    store = _store(tmp_path)
    papers = [
        make_paper(external_id="old", published=dt.datetime(2026, 6, 10, tzinfo=dt.UTC)),
        make_paper(external_id="new", published=dt.datetime(2026, 6, 20, tzinfo=dt.UTC)),
    ]
    run_acquisition(config, FakeSource(Domain.QUANTUM_MECHANICS, papers), store)
    assert store.watermark() == dt.datetime(2026, 6, 20, tzinfo=dt.UTC)


def test_future_published_date_is_capped_at_run_time(tmp_path) -> None:
    # PubMed ahead-of-print papers carry future issue dates; the watermark must
    # not jump past the run time or the next run's `since` skips everything.
    config = get_domain_config(Domain.NEUROSCIENCE)
    store = _store(tmp_path, Domain.NEUROSCIENCE)
    now = dt.datetime(2026, 7, 11, tzinfo=dt.UTC)
    future = make_paper(
        domain=Domain.NEUROSCIENCE,
        external_id="ahead-of-print",
        published=dt.datetime(2026, 12, 1, tzinfo=dt.UTC),
    )
    run_acquisition(config, FakeSource(Domain.NEUROSCIENCE, [future]), store, now=now)
    assert store.watermark() == now


def test_cold_start_empty_flag(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    store = _store(tmp_path)

    empty = run_acquisition(config, FakeSource(Domain.QUANTUM_MECHANICS, []), store)
    assert empty.cold_start_empty is True  # first run, nothing fetched

    ok = run_acquisition(
        config, FakeSource(Domain.QUANTUM_MECHANICS, [make_paper()]), store, force=True
    )
    assert ok.cold_start_empty is False  # fetched something


def test_empty_fetch_does_not_advance_watermark(tmp_path) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    store = _store(tmp_path)
    start = dt.datetime(2026, 6, 20, 12, tzinfo=dt.UTC)

    # A run that fetches nothing (e.g. a misconfigured source) must not skip
    # the window: watermark stays None, last_run still advances for cadence.
    run_acquisition(config, FakeSource(Domain.QUANTUM_MECHANICS, []), store, now=start)
    assert store.watermark() is None
    assert store.last_run() == start

    # The next run therefore still cold-starts and can recover.
    src = FakeSource(Domain.QUANTUM_MECHANICS, [make_paper()])
    run_acquisition(config, src, store, now=start + dt.timedelta(days=2), force=True)
    assert src.calls[0][0] is None  # cold start, not skipped
