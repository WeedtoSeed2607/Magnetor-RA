"""Per-domain acquisition configuration (Spec Sections 3 & 4).

This module is the single source of truth mapping each domain to its storage
directory, primary source label, acquisition mode, cadence, and trend-model
name. Nothing here reaches the network; it only describes intent.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

from magnetor.errors import ConfigError
from magnetor.types import AcquisitionMode, Domain


def _default_data_root() -> Path:
    """User-writable storage root.

    The spec's ``/data/<domain>`` is illustrative; on Windows ``/data`` resolves
    to ``C:\\data`` at the drive root, which typically needs administrator rights
    to create or write. Default instead to a per-user location that never
    requires elevation. Override with ``MAGNETOR_DATA_ROOT``.
    """
    override = os.environ.get("MAGNETOR_DATA_ROOT")
    if override:
        return Path(override)
    local_appdata = os.environ.get("LOCALAPPDATA")  # Windows
    if local_appdata:
        return Path(local_appdata) / "Magnetor" / "data"
    return Path.home() / ".magnetor" / "data"  # POSIX / fallback


#: Root under which every domain gets its own isolated subdirectory.
DATA_ROOT = _default_data_root()

#: Anchor-Lock threshold (Spec Section 7.2). The spec requires a *per-domain*
#: value calibrated against a labeled query-paper validation set; we don't have
#: that set yet, so this is a shared placeholder default each domain can override
#: in the registry below. It is logged whenever a path is selected (per the
#: spec's "logged and reviewed periodically" rule) and meant to be recalibrated
#: once labeled data exists — never treated as a settled global constant.
#: Branch A volume gate (Spec 6): re-run trend analysis once this many new
#: documents have accumulated since the last run, per domain.
_DEFAULT_TREND_MIN_NEW_DOCS = 10

#: Empirically informed placeholder for domains without observed data. From a
#: 2026-07-11 batch (n=18 forced-domain queries, voyage-4-lite): top-1 match
#: scores had overall median 0.489, Q1 0.455, Q3 0.547, max 0.596 — nowhere near
#: the removed flat 95% (Spec 7.1). The three data-bearing domains override this
#: with their own observed medians below; humanities keep this until they have a
#: corpus. This is magnitude-informed, NOT label-calibrated — recalibrate per
#: domain against a labeled query-paper set when one exists.
_DEFAULT_ANCHOR_THRESHOLD = 0.48


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Everything the pipeline needs to know about one domain, statically."""

    domain: Domain
    primary_source: str
    acquisition_mode: AcquisitionMode
    cadence: dt.timedelta
    trend_model: str
    #: Score at/above which Branch B locks a single anchor paper (Spec 7.2).
    #: Per-domain and calibratable; defaults to the shared placeholder.
    anchor_threshold: float = _DEFAULT_ANCHOR_THRESHOLD
    #: New documents since the last trend run before Branch A re-runs (Spec 6:
    #: refresh is gated on new-doc *volume* per domain, not a wall clock).
    trend_min_new_docs: int = _DEFAULT_TREND_MIN_NEW_DOCS

    @property
    def storage_dir(self) -> Path:
        """Isolated per-domain directory, e.g. ``/data/qm``.

        Derived from :data:`DATA_ROOT` at access time so overriding the root
        env var relocates every domain consistently.
        """
        return DATA_ROOT / self.domain.value


# Cadence values follow Spec Section 4: fast group is a 24h automated pull; slow
# group runs on a 7-30 day batch. This is the minimum wall-clock interval
# between runs, enforced by the pipeline's cadence gate.
# NOTE: the spec's *volume-gated* refresh (trigger on new-document count rather
# than the clock) applies to Branch A trend refresh and is a later phase; Phase 1
# acquisition gates on time only.
_DAY = dt.timedelta(days=1)

_REGISTRY: dict[Domain, DomainConfig] = {
    # Anchor thresholds below = each domain's observed top-1 median (2026-07-11
    # batch), rounded. Placeholders pending label-based calibration (Spec 7.2).
    Domain.QUANTUM_MECHANICS: DomainConfig(
        domain=Domain.QUANTUM_MECHANICS,
        primary_source="arXiv",
        acquisition_mode=AcquisitionMode.AUTOMATED_BULK,
        cadence=_DAY,
        trend_model="qm_dtm",
        anchor_threshold=0.55,  # observed median 0.546
    ),
    Domain.MATHEMATICS: DomainConfig(
        domain=Domain.MATHEMATICS,
        primary_source="arXiv",
        acquisition_mode=AcquisitionMode.AUTOMATED_BULK,
        cadence=_DAY,
        trend_model="math_dtm",
        anchor_threshold=0.47,  # observed median 0.466
    ),
    Domain.NEUROSCIENCE: DomainConfig(
        domain=Domain.NEUROSCIENCE,
        primary_source="PubMed Central",
        acquisition_mode=AcquisitionMode.AUTOMATED_BULK,
        cadence=_DAY,
        trend_model="neuro_dtm",
        anchor_threshold=0.47,  # observed median 0.471
    ),
    Domain.PHILOSOPHY: DomainConfig(
        domain=Domain.PHILOSOPHY,
        primary_source="PhilPapers OA subset + OpenAlex/CORE",
        acquisition_mode=AcquisitionMode.BATCH_MANUAL,
        cadence=dt.timedelta(days=7),
        trend_model="phil_dtm",
    ),
    Domain.ANTHROPOLOGY: DomainConfig(
        domain=Domain.ANTHROPOLOGY,
        primary_source="AnthroSource + OpenAlex",
        acquisition_mode=AcquisitionMode.BATCH_MANUAL,
        cadence=dt.timedelta(days=30),
        trend_model="anthro_dtm",
    ),
    Domain.HISTORY: DomainConfig(
        domain=Domain.HISTORY,
        primary_source="JSTOR (Data for Research) + OpenAlex",
        acquisition_mode=AcquisitionMode.BATCH_MANUAL,
        cadence=dt.timedelta(days=30),
        trend_model="hist_dtm",
    ),
}


def get_domain_config(domain: Domain) -> DomainConfig:
    """Return the static config for ``domain``.

    Raises:
        ConfigError: No config is registered for the domain (should be
            unreachable while :class:`Domain` and ``_REGISTRY`` stay in sync;
            guarded so a future enum addition fails loudly rather than silently).
    """
    try:
        return _REGISTRY[domain]
    except KeyError as exc:
        raise ConfigError(f"No acquisition config registered for domain {domain!r}") from exc


def all_domains() -> tuple[Domain, ...]:
    """All configured domains, in registry order."""
    return tuple(_REGISTRY)


def global_store_path(name: str) -> Path:
    """Path to a global (cross-domain) artifact under the data root.

    Read at call time so a patched :data:`DATA_ROOT` is honoured. Used for
    cross-domain artifacts the isolation rule permits at the router layer (e.g.
    the routing-decision log), never for document bodies or embeddings.
    """
    return DATA_ROOT / name


def resolve_domain(name: str) -> Domain:
    """Map a CLI token (e.g. ``qm``, ``neuro``) to a :class:`Domain`.

    Raises:
        ConfigError: The token matches no domain value.
    """
    try:
        return Domain(name)
    except ValueError as exc:
        known = ", ".join(d.value for d in all_domains())
        raise ConfigError(f"Unknown domain {name!r}; expected one of: {known}") from exc
