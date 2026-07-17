"""Manual/batch acquisition source (Philosophy, Anthropology, History).

The slow-group domains have no real-time bulk API (Spec Section 1, FEASIBILITY):
JSTOR Data for Research is a mediated multi-day request service and PhilPapers'
full text is restricted. So acquisition here is operator-driven: a human runs
the mediated request or aggregator query, drops the resulting metadata records
into the domain's ``_inbox`` directory as JSON, and this source ingests them.

Redistribution guard (Spec Section 4): metadata is stored by default; full text
is accepted only when the drop record carries an explicit ``license``. A record
that claims full text without a license raises :class:`RedistributionError`
rather than being stored, forcing the operator to confirm permitted use.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path

from magnetor.errors import ParseError, RedistributionError
from magnetor.types import Domain, Paper

INBOX_DIRNAME = "_inbox"
ARCHIVE_DIRNAME = "_archive"

# The three domains this source is allowed to serve.
_ALLOWED = frozenset(
    {Domain.PHILOSOPHY, Domain.ANTHROPOLOGY, Domain.HISTORY}
)


class ManualBatchSource:
    """Ingest operator-supplied metadata drops for one slow-group domain."""

    def __init__(self, domain: Domain, inbox_dir: Path) -> None:
        if domain not in _ALLOWED:
            raise ParseError(f"ManualBatchSource does not serve domain {domain!r}")
        self._domain = domain
        self._inbox = inbox_dir
        #: Drop files consumed in the most recent fetch, archived on run success.
        self._consumed: list[Path] = []

    @property
    def domain(self) -> Domain:
        return self._domain

    def fetch(self, *, since: dt.datetime | None, limit: int) -> Iterable[Paper]:
        """Read drop files from the inbox, in filename order, up to ``limit``.

        ``since`` is honoured against each record's ``published`` date when
        present; records without a date are always included (the operator
        curated them deliberately). The nested ``_archive`` subdir is skipped.
        """
        self._consumed = []
        if not self._inbox.exists():
            return []
        papers: list[Paper] = []
        for path in sorted(self._inbox.glob("*.json")):
            if len(papers) >= limit:
                break
            paper = self._load_drop(path)
            self._consumed.append(path)
            if since is not None and paper.published is not None and paper.published < since:
                continue
            papers.append(paper)
        return papers

    def on_run_complete(self) -> None:
        """Move successfully-processed drop files into ``_inbox/_archive``.

        Keeps the inbox from growing unbounded and stops each run re-parsing the
        same drops. Called by the pipeline only after records are persisted, so
        a failed run leaves the inbox untouched for retry.
        """
        if not self._consumed:
            return
        archive = self._inbox / ARCHIVE_DIRNAME
        archive.mkdir(parents=True, exist_ok=True)
        for path in self._consumed:
            if path.exists():
                path.replace(archive / path.name)
        self._consumed = []

    def _load_drop(self, path: Path) -> Paper:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ParseError(f"Malformed drop file {path.name}: {exc}") from exc
        if not isinstance(record, dict):
            raise ParseError(f"Drop file {path.name} is not a JSON object")

        claims_full_text = bool(record.get("full_text_available"))
        license_ = record.get("license")
        if claims_full_text and not license_:
            raise RedistributionError(
                f"{path.name} claims full text without a license; confirm "
                f"permitted use with the source before storing (Spec Section 4)"
            )

        external_id = str(record.get("external_id") or record.get("doi") or path.stem)
        return Paper(
            domain=self._domain,
            source=str(record.get("source", "manual_batch")),
            external_id=external_id,
            title=_clean(str(record.get("title", ""))),
            abstract=_clean(str(record.get("abstract", ""))),
            authors=tuple(str(a) for a in record.get("authors", [])),
            published=_parse_dt(record.get("published")),
            doi=(str(record["doi"]) if record.get("doi") else None),
            pdf_url=(str(record["pdf_url"]) if record.get("pdf_url") else None),
            full_text_available=claims_full_text and bool(license_),
            license=(str(license_) if license_ else None),
        )


def _clean(value: str) -> str:
    return " ".join(value.split())


def _parse_dt(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Date-only drops (e.g. "2026-06-10") parse as naive; force UTC so they
    # compare cleanly against the timezone-aware `since` bound in the pipeline.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
