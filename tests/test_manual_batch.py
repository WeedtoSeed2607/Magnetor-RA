"""ManualBatchSource: drop-file ingestion and redistribution guard."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from magnetor.errors import ParseError, RedistributionError
from magnetor.sources.manual_batch import ARCHIVE_DIRNAME, ManualBatchSource
from magnetor.types import Domain


def _drop(inbox: Path, name: str, record: dict[str, object]) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(json.dumps(record), encoding="utf-8")


def test_metadata_only_by_default(tmp_path) -> None:
    inbox = tmp_path / "inbox"
    _drop(
        inbox,
        "a.json",
        {
            "external_id": "phil-001",
            "title": "On Being",
            "abstract": "A treatise.",
            "authors": ["Aristotle"],
            "published": "2026-06-10",
        },
    )
    source = ManualBatchSource(Domain.PHILOSOPHY, inbox)
    papers = list(source.fetch(since=None, limit=10))

    assert len(papers) == 1
    assert papers[0].full_text_available is False
    assert papers[0].license is None
    assert papers[0].published == dt.datetime(2026, 6, 10, tzinfo=dt.UTC)


def test_full_text_with_license_is_allowed(tmp_path) -> None:
    inbox = tmp_path / "inbox"
    _drop(
        inbox,
        "b.json",
        {
            "external_id": "hist-002",
            "title": "A History",
            "full_text_available": True,
            "license": "CC-BY-4.0",
        },
    )
    source = ManualBatchSource(Domain.HISTORY, inbox)
    papers = list(source.fetch(since=None, limit=10))
    assert papers[0].full_text_available is True
    assert papers[0].license == "CC-BY-4.0"


def test_full_text_without_license_is_refused(tmp_path) -> None:
    inbox = tmp_path / "inbox"
    _drop(
        inbox,
        "c.json",
        {"external_id": "anthro-003", "title": "Kinship", "full_text_available": True},
    )
    source = ManualBatchSource(Domain.ANTHROPOLOGY, inbox)
    with pytest.raises(RedistributionError):
        list(source.fetch(since=None, limit=10))


def test_missing_inbox_returns_empty(tmp_path) -> None:
    source = ManualBatchSource(Domain.PHILOSOPHY, tmp_path / "nope")
    assert list(source.fetch(since=None, limit=10)) == []


def test_rejects_unsupported_domain(tmp_path) -> None:
    with pytest.raises(ParseError):
        ManualBatchSource(Domain.QUANTUM_MECHANICS, tmp_path)


def test_on_run_complete_archives_processed_drops(tmp_path) -> None:
    inbox = tmp_path / "inbox"
    _drop(inbox, "a.json", {"external_id": "p-1", "title": "One"})
    _drop(inbox, "b.json", {"external_id": "p-2", "title": "Two"})
    source = ManualBatchSource(Domain.PHILOSOPHY, inbox)

    papers = list(source.fetch(since=None, limit=10))
    assert len(papers) == 2
    # Before the hook, drops are still in the inbox.
    assert sorted(p.name for p in inbox.glob("*.json")) == ["a.json", "b.json"]

    source.on_run_complete()
    # After: inbox is drained, files live under _archive, and re-fetch is empty.
    assert list(inbox.glob("*.json")) == []
    archived = sorted(p.name for p in (inbox / ARCHIVE_DIRNAME).glob("*.json"))
    assert archived == ["a.json", "b.json"]
    assert list(source.fetch(since=None, limit=10)) == []


def test_archive_subdir_is_not_reingested(tmp_path) -> None:
    inbox = tmp_path / "inbox"
    _drop(inbox, "a.json", {"external_id": "p-1", "title": "One"})
    source = ManualBatchSource(Domain.HISTORY, inbox)
    list(source.fetch(since=None, limit=10))
    source.on_run_complete()
    # A second, fresh drop is picked up; the archived one is not.
    _drop(inbox, "b.json", {"external_id": "p-2", "title": "Two"})
    papers = list(source.fetch(since=None, limit=10))
    assert [p.external_id for p in papers] == ["p-2"]
