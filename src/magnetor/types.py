"""Core data shapes for the acquisition layer.

Plain frozen dataclasses, deliberately: the acquired records have a simple,
stable shape and no behaviour, so a class hierarchy or a validation library
would be premature (escalation ladder: plain code first). Validation lives at
the source boundary where external data enters, not on these carriers.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum


class Domain(StrEnum):
    """The six target domains. Value doubles as the on-disk directory name."""

    QUANTUM_MECHANICS = "qm"
    MATHEMATICS = "math"
    NEUROSCIENCE = "neuro"
    PHILOSOPHY = "philosophy"
    ANTHROPOLOGY = "anthro"
    HISTORY = "history"


class AcquisitionMode(StrEnum):
    """How a domain is ingested (Spec Section 4)."""

    #: Fast group — QM/Math/Neuro: direct daily bulk API pull, full text.
    AUTOMATED_BULK = "automated_bulk"
    #: Slow group — Philosophy/Anthro/History: mediated/manual batch, metadata
    #: + n-grams by default, full text only where licensing permits.
    BATCH_MANUAL = "batch_manual"


@dataclass(frozen=True, slots=True)
class Paper:
    """A single acquired record.

    This is the raw acquisition artifact — metadata plus, where terms allow, a
    pointer to full text. It is intentionally source-agnostic so the pipeline
    and storage layers never branch on which domain produced it.
    """

    domain: Domain
    source: str
    external_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: dt.datetime | None
    updated: dt.datetime | None = None
    doi: str | None = None
    pdf_url: str | None = None
    full_text_available: bool = False
    license: str | None = None
    #: Verbatim source-specific fields we did not lift into typed attributes.
    raw: dict[str, str] = field(default_factory=dict, compare=False)

    def dedup_key(self) -> str:
        """Stable identity for within-domain idempotency.

        Prefers DOI (globally unique across sources), falling back to the
        source-local external id. Cross-*domain* dedup is a later storage-layer
        concern (Spec Section 10 ``dedup_index.json``) and is out of Phase 1
        scope; this only prevents re-storing the same paper within one domain.
        """
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return f"{self.source}:{self.external_id}"


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Outcome of one pipeline run for one domain."""

    domain: Domain
    fetched: int
    stored: int
    skipped_duplicates: int
    ran: bool
    reason: str = ""
    #: True when a first-ever (cold-start) run fetched nothing — usually a sign
    #: of a misconfigured source rather than a genuinely quiet feed.
    cold_start_empty: bool = False
