"""Storage and run-state for the acquisition layer.

Enforces the load-bearing invariant from Spec Section 3: a store bound to one
domain physically cannot write outside that domain's directory. Every path is
resolved and checked to sit under the domain root before any write.

Phase 1 persists metadata records and per-domain run state as JSON. The vector
index and ``dedup_index.json`` (Spec Section 10) are later-phase concerns and
are intentionally absent here.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
from pathlib import Path

from magnetor.errors import MagnetorError
from magnetor.types import Domain, Paper

_STATE_FILENAME = "_state.json"
_RECORDS_DIRNAME = "records"


class DomainStore:
    """Filesystem-backed store scoped to exactly one domain directory."""

    def __init__(self, domain: Domain, storage_dir: Path) -> None:
        self._domain = domain
        # Logical absolute path, NOT Path.resolve(): the isolation check is
        # structural, and resolve() canonicalises symlinks/junctions and
        # filesystem redirection (e.g. MSIX-packaged apps redirect
        # %LOCALAPPDATA% to a per-package LocalCache), which would make a
        # legitimately-nested record path look "outside" the root.
        self._root = Path(os.path.abspath(storage_dir))

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def root(self) -> Path:
        return self._root

    def _guard(self, path: Path) -> Path:
        """Reject any path that escapes this domain's root (isolation guard).

        Traversal via crafted filenames is already prevented by
        :func:`_safe_filename`; this is a structural belt-and-suspenders check
        on the logical (unresolved) absolute path.
        """
        candidate = Path(os.path.abspath(path))
        root_cmp = os.path.normcase(str(self._root))
        cand_cmp = os.path.normcase(str(candidate))
        if cand_cmp != root_cmp and not cand_cmp.startswith(root_cmp + os.sep):
            raise MagnetorError(
                f"Isolation violation: {candidate} is outside domain root {self._root}"
            )
        return candidate

    # --- run state -----------------------------------------------------------

    def _state_path(self) -> Path:
        return self._guard(self._root / _STATE_FILENAME)

    def load_state(self) -> dict[str, str]:
        """Return persisted run state, or an empty dict on first run."""
        path = self._state_path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def last_run(self) -> dt.datetime | None:
        """When a run was last *attempted* (drives the cadence gate).

        Advances on every recorded run, even one that fetched nothing, so the
        scheduler doesn't hammer a quiet source.
        """
        raw = self.load_state().get("last_run")
        return dt.datetime.fromisoformat(raw) if raw else None

    def watermark(self) -> dt.datetime | None:
        """Newest publication time covered so far (drives the ``since`` filter).

        Distinct from :meth:`last_run`: it advances only to papers actually
        seen, so a failed or empty fetch never skips an uncovered window.
        """
        raw = self.load_state().get("watermark")
        return dt.datetime.fromisoformat(raw) if raw else None

    def record_run(
        self,
        when: dt.datetime,
        *,
        stored: int,
        watermark: dt.datetime | None = None,
    ) -> None:
        """Persist a completed run.

        ``when`` always advances the cadence timestamp. ``watermark`` advances
        the ingestion high-water mark, but only forward and only when a real
        publication time was seen — an empty fetch (``watermark=None``) leaves
        the mark untouched so the next run re-covers that window.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        state = self.load_state()
        state["last_run"] = when.isoformat()
        state["last_stored"] = str(stored)
        if watermark is not None:
            current = self.watermark()
            if current is None or watermark > current:
                state["watermark"] = watermark.isoformat()
        self._state_path().write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    def last_trend_count(self) -> int:
        """Record count as of the last Branch-A trend run (0 if never run)."""
        raw = self.load_state().get("last_trend_count")
        return int(raw) if raw else 0

    def last_trend_run(self) -> dt.datetime | None:
        """When Branch-A trend analysis last ran for this domain."""
        raw = self.load_state().get("last_trend_run")
        return dt.datetime.fromisoformat(raw) if raw else None

    def record_trend_run(self, when: dt.datetime, *, record_count: int) -> None:
        """Persist that a trend run completed with ``record_count`` docs present."""
        self._root.mkdir(parents=True, exist_ok=True)
        state = self.load_state()
        state["last_trend_run"] = when.isoformat()
        state["last_trend_count"] = str(record_count)
        self._state_path().write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    # --- records -------------------------------------------------------------

    def _records_dir(self) -> Path:
        return self._guard(self._root / _RECORDS_DIRNAME)

    def known_keys(self) -> set[str]:
        """Dedup keys already stored for this domain (within-domain idempotency)."""
        records_dir = self._records_dir()
        if not records_dir.exists():
            return set()
        return {p.stem for p in records_dir.glob("*.json")}

    def record_count(self) -> int:
        """Number of stored records, without loading them."""
        records_dir = self._records_dir()
        if not records_dir.exists():
            return 0
        return sum(1 for _ in records_dir.glob("*.json"))

    def store(self, paper: Paper) -> bool:
        """Persist one paper's metadata; return ``False`` if already stored.

        Guards that the paper belongs to this store's domain — a defensive
        restatement of the isolation rule at the record level.
        """
        if paper.domain is not self._domain:
            raise MagnetorError(
                f"Refusing to store {paper.domain.value!r} paper in "
                f"{self._domain.value!r} store"
            )
        records_dir = self._records_dir()
        records_dir.mkdir(parents=True, exist_ok=True)
        key = _safe_filename(paper.dedup_key())
        target = self._guard(records_dir / f"{key}.json")
        if target.exists():
            return False
        target.write_text(_paper_to_json(paper), encoding="utf-8")
        return True

    def read_records(self, *, limit: int | None = None) -> list[Paper]:
        """Load stored papers, newest first. Empty if nothing acquired yet."""
        records_dir = self._records_dir()
        if not records_dir.exists():
            return []
        papers = [
            _paper_from_json(path.read_text(encoding="utf-8"))
            for path in records_dir.glob("*.json")
        ]
        papers.sort(key=_published_sort_key, reverse=True)
        return papers if limit is None else papers[:limit]


def _safe_filename(key: str) -> str:
    """Make a dedup key safe to use as a filename stem."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in key)


def _published_sort_key(paper: Paper) -> dt.datetime:
    """Sort key placing dated papers newest-first and undated ones last."""
    return paper.published or dt.datetime.min.replace(tzinfo=dt.UTC)


def _paper_to_json(paper: Paper) -> str:
    data = dataclasses.asdict(paper)
    data["domain"] = paper.domain.value
    data["published"] = paper.published.isoformat() if paper.published else None
    data["updated"] = paper.updated.isoformat() if paper.updated else None
    data["authors"] = list(paper.authors)
    return json.dumps(data, indent=2, sort_keys=True)


def _paper_from_json(text: str) -> Paper:
    """Reconstruct a :class:`Paper` from the JSON written by ``_paper_to_json``."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise MagnetorError("stored record is not a JSON object")
    raw = data.get("raw")
    return Paper(
        domain=Domain(str(data["domain"])),
        source=str(data.get("source", "")),
        external_id=str(data.get("external_id", "")),
        title=str(data.get("title", "")),
        abstract=str(data.get("abstract", "")),
        authors=tuple(str(a) for a in _as_list(data.get("authors"))),
        published=_from_iso(data.get("published")),
        updated=_from_iso(data.get("updated")),
        doi=_opt_str(data.get("doi")),
        pdf_url=_opt_str(data.get("pdf_url")),
        full_text_available=bool(data.get("full_text_available", False)),
        license=_opt_str(data.get("license")),
        raw={str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {},
    )


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _opt_str(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def _from_iso(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None
