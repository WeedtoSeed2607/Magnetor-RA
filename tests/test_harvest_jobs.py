"""Branch C — background harvest jobs launched from the dashboard (ADR-0006 I5)."""

from __future__ import annotations

import subprocess
from typing import ClassVar

from magnetor import harvest_jobs
from magnetor.graph import query_hash
from magnetor.harvest_jobs import (
    DONE,
    ENABLE_ENV,
    FAILED,
    RUNNING,
    harvest_ui_enabled,
    job_for,
    log_tail,
    recent_jobs,
    start_anchor,
    start_harvest,
)


class _FakePopen:
    """Records the argv instead of spawning — a real harvest would hit the network."""

    calls: ClassVar[list[list[str]]] = []

    def __init__(self, argv, **kwargs) -> None:
        type(self).calls.append(list(argv))
        self.kwargs = kwargs


def _no_spawn(monkeypatch) -> type[_FakePopen]:
    _FakePopen.calls = []
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    return _FakePopen


def test_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert harvest_ui_enabled() is False


def test_falsey_values_keep_it_disabled(monkeypatch) -> None:
    for value in ("", "0", "false", "no", "  "):
        monkeypatch.setenv(ENABLE_ENV, value)
        assert harvest_ui_enabled() is False
    monkeypatch.setenv(ENABLE_ENV, "1")
    assert harvest_ui_enabled() is True


def test_start_harvest_spawns_the_cli_with_the_query_and_knobs(monkeypatch, tmp_path) -> None:
    fake = _no_spawn(monkeypatch)
    job = start_harvest("quantum error correction", limit=120, expand=1, jobs_dir=tmp_path)

    assert job.status == RUNNING
    assert job.hash == query_hash("quantum error correction")
    argv = fake.calls[0]
    assert "harvest" in argv
    assert "quantum error correction" in argv
    assert argv[argv.index("--limit") + 1] == "120"
    assert argv[argv.index("--expand") + 1] == "1"
    # The query is passed as its own argv entry, never interpolated into a shell
    # string, so a question containing quotes or "&" cannot become a command.
    assert not any(part.startswith("&") for part in argv)


def test_start_anchor_spawns_the_anchor_subcommand(monkeypatch, tmp_path) -> None:
    fake = _no_spawn(monkeypatch)
    job = start_anchor("10.1037/abc", limit=80, backward=1, forward_fanout=3,
                       jobs_dir=tmp_path)
    argv = fake.calls[0]
    assert "anchor" in argv and "harvest" not in argv
    assert "10.1037/abc" in argv
    assert argv[argv.index("--backward") + 1] == "1"
    assert argv[argv.index("--forward-fanout") + 1] == "3"
    assert job.status == RUNNING


def test_anchor_and_harvest_jobs_are_tracked_the_same_way(monkeypatch, tmp_path) -> None:
    _no_spawn(monkeypatch)
    start_anchor("10.1037/abc", jobs_dir=tmp_path)
    found = job_for("10.1037/abc", jobs_dir=tmp_path)
    assert found is not None and found.status == RUNNING


def test_started_job_is_running_until_the_marker_appears(monkeypatch, tmp_path) -> None:
    _no_spawn(monkeypatch)
    start_harvest("q", jobs_dir=tmp_path)
    found = job_for("q", jobs_dir=tmp_path)
    assert found is not None
    assert found.status == RUNNING
    assert found.exit_code is None


def test_exit_marker_resolves_success_and_failure(monkeypatch, tmp_path) -> None:
    _no_spawn(monkeypatch)
    start_harvest("q", jobs_dir=tmp_path)
    marker = tmp_path / f"{query_hash('q')}.exit"

    marker.write_text("0", encoding="utf-8")
    ok = job_for("q", jobs_dir=tmp_path)
    assert ok is not None and ok.status == DONE and ok.exit_code == 0

    marker.write_text("2", encoding="utf-8")
    bad = job_for("q", jobs_dir=tmp_path)
    assert bad is not None and bad.status == FAILED and bad.exit_code == 2


def test_restarting_clears_a_previous_verdict(monkeypatch, tmp_path) -> None:
    _no_spawn(monkeypatch)
    start_harvest("q", jobs_dir=tmp_path)
    (tmp_path / f"{query_hash('q')}.exit").write_text("2", encoding="utf-8")
    assert job_for("q", jobs_dir=tmp_path).status == FAILED  # type: ignore[union-attr]

    start_harvest("q", jobs_dir=tmp_path)  # a rerun must not inherit the old failure
    assert job_for("q", jobs_dir=tmp_path).status == RUNNING  # type: ignore[union-attr]


def test_job_for_unknown_query_is_none(tmp_path) -> None:
    assert job_for("never launched", jobs_dir=tmp_path) is None


def test_log_tail_returns_last_nonblank_lines(monkeypatch, tmp_path) -> None:
    _no_spawn(monkeypatch)
    job = start_harvest("q", jobs_dir=tmp_path)
    job.log_path.write_text("one\n\ntwo\nthree\n", encoding="utf-8")
    assert log_tail(job, lines=2) == ["two", "three"]


def test_recent_jobs_are_newest_first(monkeypatch, tmp_path) -> None:
    _no_spawn(monkeypatch)
    start_harvest("older", jobs_dir=tmp_path)
    start_harvest("newer", jobs_dir=tmp_path)
    # Rewrite timestamps so ordering is deterministic rather than clock-dependent.
    (tmp_path / f"{query_hash('older')}.json").write_text(
        '{"query": "older", "started_at": "2020-01-01T00:00:00+00:00"}', encoding="utf-8"
    )
    (tmp_path / f"{query_hash('newer')}.json").write_text(
        '{"query": "newer", "started_at": "2030-01-01T00:00:00+00:00"}', encoding="utf-8"
    )
    assert [j.query for j in recent_jobs(jobs_dir=tmp_path)] == ["newer", "older"]


def test_corrupt_record_is_skipped_not_crashed(tmp_path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert recent_jobs(jobs_dir=tmp_path) == []


def test_jobs_live_outside_the_graphs_directory() -> None:
    """A job record must never be globbed up as if it were a graph document."""
    assert harvest_jobs.JOBS_DIRNAME != "graphs"
