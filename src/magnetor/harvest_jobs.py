"""Branch C — starting a harvest from the dashboard without breaking I5.

ADR-0006 is explicit that graph construction **must not** run inside a dashboard
request cycle (D1's consequence, and invariant I5): a harvest is minutes to tens
of minutes, so doing it inline would block the UI and exceed hosting timeouts.

This module removes the need to drop to a terminal *without* violating that. The
page **spawns** the existing ``magnetor harvest`` CLI as a detached child process
and thereafter only ever **reads** files — the job's log, its exit marker, and
finally the persisted graph. Compute stays offline and the request stays short,
which is exactly what I5 protects.

The child is wrapped in a one-line runner that records the CLI's exit status, so
completion is known exactly rather than guessed. Polling by artifact alone cannot
distinguish "still running" from "died on startup", and a PID liveness check is
not portable — on Windows ``os.kill(pid, 0)`` terminates the target rather than
probing it.

Gated by ``MAGNETOR_ENABLE_HARVEST_UI``. The hosted entry point never sets it, so
a public deployment cannot spawn processes or spend API quota; the local
launcher (``mag.cmd``) does.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from magnetor.config import global_store_path
from magnetor.graph import query_hash

JOBS_DIRNAME = "harvest_jobs"

#: Opt-in flag. Default-deny: spawning processes is not something a hosted,
#: publicly-reachable page should ever do by accident.
ENABLE_ENV = "MAGNETOR_ENABLE_HARVEST_UI"

RUNNING = "running"
DONE = "done"
FAILED = "failed"

# Runs the real CLI, then writes its exit code where the dashboard can read it.
# argv: [marker_path, *command]
_WRAPPER = (
    "import subprocess,sys,pathlib;"
    "rc=subprocess.call(sys.argv[2:]);"
    "pathlib.Path(sys.argv[1]).write_text(str(rc),encoding='utf-8')"
)


@dataclass(frozen=True, slots=True)
class HarvestJob:
    query: str
    hash: str
    started_at: str
    status: str  # RUNNING | DONE | FAILED
    exit_code: int | None
    log_path: Path


def harvest_ui_enabled() -> bool:
    """Whether this deployment may launch harvests from the page."""
    return os.environ.get(ENABLE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _jobs_root(jobs_dir: Path | None) -> Path:
    return jobs_dir or global_store_path(JOBS_DIRNAME)


def start_harvest(
    query: str,
    *,
    limit: int = 200,
    expand: int = 2,
    expand_top: int = 60,
    resamples: int = 1000,
    top_n: int = 60,
    jobs_dir: Path | None = None,
) -> HarvestJob:
    """Spawn ``magnetor harvest`` in the background and record the job.

    Returns immediately; the caller polls :func:`job_for`. The child inherits the
    current environment, so ``MAGNETOR_DATA_ROOT`` and any keys loaded from
    ``.env`` reach it unchanged and the graph lands where the dashboard reads.
    """
    root = _jobs_root(jobs_dir)
    root.mkdir(parents=True, exist_ok=True)
    digest = query_hash(query)
    log_path = root / f"{digest}.log"
    marker = root / f"{digest}.exit"
    marker.unlink(missing_ok=True)  # a rerun must not inherit the old verdict

    command = [
        sys.executable, "-m", "magnetor.cli", "harvest", query,
        "--limit", str(limit),
        "--expand", str(expand),
        "--expand-top", str(expand_top),
        "--resamples", str(resamples),
        "--top-n", str(top_n),
    ]
    started_at = dt.datetime.now(dt.UTC).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        # argv list, never a shell string: a query containing quotes, "&" or "|"
        # is data to the child, not syntax.
        subprocess.Popen(
            [sys.executable, "-c", _WRAPPER, str(marker), *command],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Keep the child off the desktop; harmless where the flag is absent.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    (root / f"{digest}.json").write_text(
        json.dumps({"query": query, "started_at": started_at}, indent=2),
        encoding="utf-8",
    )
    return HarvestJob(
        query=query, hash=digest, started_at=started_at,
        status=RUNNING, exit_code=None, log_path=log_path,
    )


def job_for(query: str, *, jobs_dir: Path | None = None) -> HarvestJob | None:
    """The recorded job for a query, with its status resolved, or ``None``."""
    root = _jobs_root(jobs_dir)
    record = root / f"{query_hash(query)}.json"
    if not record.exists():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _resolve(str(data.get("query", query)), str(data.get("started_at", "")), root)


def _resolve(query: str, started_at: str, root: Path) -> HarvestJob:
    digest = query_hash(query)
    marker = root / f"{digest}.exit"
    exit_code: int | None = None
    status = RUNNING
    if marker.exists():
        try:
            exit_code = int(marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            exit_code = None
        status = DONE if exit_code == 0 else FAILED
    return HarvestJob(
        query=query, hash=digest, started_at=started_at,
        status=status, exit_code=exit_code, log_path=root / f"{digest}.log",
    )


def log_tail(job: HarvestJob, *, lines: int = 12) -> list[str]:
    """Last lines of a job's output, for progress and for diagnosing a failure."""
    if not job.log_path.exists():
        return []
    try:
        text = job.log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()][-lines:]


def recent_jobs(*, jobs_dir: Path | None = None, limit: int = 10) -> list[HarvestJob]:
    """Recorded jobs, newest first — the page's history of what it launched."""
    root = _jobs_root(jobs_dir)
    if not root.exists():
        return []
    jobs: list[HarvestJob] = []
    for record in root.glob("*.json"):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("query"):
            jobs.append(_resolve(str(data["query"]), str(data.get("started_at", "")), root))
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return jobs[:limit]
