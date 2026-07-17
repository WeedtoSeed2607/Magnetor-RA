"""DomainStore: isolation guard, dedup, and run-state persistence."""

from __future__ import annotations

import datetime as dt

import pytest

from magnetor.errors import MagnetorError
from magnetor.resources import DomainStore
from magnetor.types import Domain
from tests.conftest import make_paper


def _store(tmp_path, domain=Domain.QUANTUM_MECHANICS) -> DomainStore:
    return DomainStore(domain, tmp_path / "data" / domain.value)


def test_store_and_dedup(tmp_path) -> None:
    store = _store(tmp_path)
    paper = make_paper(external_id="2601.00001")
    assert store.store(paper) is True
    # Second store of the same key is a no-op.
    assert store.store(paper) is False
    assert len(store.known_keys()) == 1


def test_store_rejects_foreign_domain(tmp_path) -> None:
    store = _store(tmp_path, Domain.QUANTUM_MECHANICS)
    foreign = make_paper(domain=Domain.MATHEMATICS)
    with pytest.raises(MagnetorError):
        store.store(foreign)


def test_run_state_roundtrip(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.last_run() is None
    moment = dt.datetime(2026, 6, 20, 12, tzinfo=dt.UTC)
    store.record_run(moment, stored=3)
    assert store.last_run() == moment


def test_known_keys_survive_new_instance(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="2601.00002"))
    reopened = _store(tmp_path)
    assert len(reopened.known_keys()) == 1


def test_guard_accepts_nested_and_rejects_escape(tmp_path) -> None:
    store = _store(tmp_path)
    # A legitimately nested path under the domain root is accepted.
    assert store._guard(store.root / "records" / "x.json").name == "x.json"
    # A sibling outside the domain root is rejected.
    with pytest.raises(MagnetorError):
        store._guard(store.root.parent / "other" / "x.json")


def test_store_writes_under_root_across_normal_paths(tmp_path) -> None:
    # Full write path exercises the guard end-to-end (regression for the
    # resolve()-vs-abspath isolation false positive).
    store = _store(tmp_path)
    assert store.store(make_paper(external_id="2601.12345")) is True
    assert len(store.known_keys()) == 1


def test_read_records_roundtrips_fields(tmp_path) -> None:
    store = _store(tmp_path)
    original = make_paper(external_id="2601.55555", doi="10.1/abc")
    store.store(original)

    (loaded,) = store.read_records()
    assert loaded.external_id == original.external_id
    assert loaded.title == original.title
    assert loaded.abstract == original.abstract
    assert loaded.authors == original.authors
    assert loaded.published == original.published
    assert loaded.doi == original.doi
    assert loaded.domain is Domain.QUANTUM_MECHANICS


def test_read_records_newest_first_and_limited(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="old", published=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)))
    store.store(make_paper(external_id="new", published=dt.datetime(2026, 6, 1, tzinfo=dt.UTC)))

    newest_first = store.read_records()
    assert [p.external_id for p in newest_first] == ["new", "old"]
    assert [p.external_id for p in store.read_records(limit=1)] == ["new"]


def test_read_records_empty_when_nothing_stored(tmp_path) -> None:
    assert _store(tmp_path).read_records() == []
