"""Dependency boundaries.

External systems (arXiv, PubMed Central, manual drop folders) are reached only
through the :class:`DomainSource` protocol, so the pipeline never imports a
concrete client and tests can substitute a fake without touching the network
(baseline: "wrap external systems behind interfaces so tests can mock them").
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from magnetor.types import Domain, Paper


@runtime_checkable
class DomainSource(Protocol):
    """Contract every acquisition source implements for exactly one domain."""

    @property
    def domain(self) -> Domain:
        """The single domain this source feeds. A source never crosses domains."""
        ...

    def fetch(self, *, since: dt.datetime | None, limit: int) -> Iterable[Paper]:
        """Yield papers published/updated at or after ``since``.

        Args:
            since: Lower bound from the last successful run, or ``None`` for a
                cold start (source decides its own cold-start window).
            limit: Hard cap on records returned this run.

        Raises:
            SourceUnavailableError: The upstream could not be reached.
            ParseError: A response arrived but could not be parsed.
        """
        ...


@runtime_checkable
class SupportsPostRun(Protocol):
    """Optional hook a source may implement to clean up after a successful run.

    The pipeline calls :meth:`on_run_complete` only after records were persisted
    and run state recorded, so a source can, e.g., archive processed inputs.
    """

    def on_run_complete(self) -> None:
        """React to a completed, persisted acquisition run."""
        ...
