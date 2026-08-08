"""Branch C — the Lakatosian coding instrument (Part III of the critique document).

**What this is, and what it deliberately is not.** Part IV of the source document
states the position plainly: *"The corpus is unbuilt… the largest single piece of
unfunded work"* and *"No coding manual has been piloted."* So this builds the
**instrument** — a schema, a store, and the reliability statistic that decides
whether the schema works — not the study. Most Part III fields need full text and
human judgement (``use_novel``, ``excess_content``, ``power_regime``,
``analytic_dof``); nothing here fabricates them.

**Why an instrument is worth building before the corpus.** Section II.1 makes the
argument: *"A grammar becomes empirical the moment it is used to code a corpus."*
If independent coders cannot agree on ``level_sense`` or ``interp_schema_type``,
that is a real, publishable negative result about the framework — and III.6 is
explicit that a field below the reliability floor is *"a finding about the
framework, not a coding failure to be patched."* Krippendorff's alpha is therefore
the point of this module, not an accessory to it.

**Prefilled fields never assert more than was checked.** ``integrity_status``
prefills to ``retracted`` when OpenAlex flags it and otherwise to ``not-checked``,
never to ``clean``: Retraction Watch and PubPeer are not consulted here, and
recording an unchecked paper as clean would manufacture the very false precision
the architecture exists to avoid (C3 — absence is displayed, not imputed).

**Storage is append-only JSONL, one file per graph.** Coding is evidence about
coders as much as about papers, so an overwrite would destroy the disagreement
that alpha is computed from.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from magnetor.config import global_store_path
from magnetor.graph import query_hash

CODING_DIRNAME = "coding"

#: Environment override for who is coding. III.6 requires two or more independent
#: coders, so an unattributed record cannot contribute to reliability.
CODER_ENV = "MAGNETOR_CODER"

UNSET = "unset"

#: The Part III vocabularies. ``unset`` is always permitted and always the
#: default: a coder who has not judged a field must be distinguishable from one
#: who judged it and found nothing, which is the same reason the manual carries
#: ``none-of-the-above`` and ``unspecified`` as real values rather than blanks.
FIELDS: dict[str, tuple[str, ...]] = {
    "explanation_type": ("descriptive", "mechanistic", "normative", "none-of-the-above"),
    "level_sense": ("L_scale", "L_comp", "L_const", "L_ideal", "L_dyn", "unspecified"),
    "interp_schema_type": ("T0", "T1", "T2", "T3"),
    "interp_invertible": ("yes", "no", "not-stated"),
    "data_or_phenomenon": ("datum-claim", "phenomenon-claim"),
    "normative_subscript": ("N_sel", "N_asc", "N_fit", "not-applicable"),
    "normative_presented_as": ("N_sel", "N_asc", "N_fit", "not-applicable"),
    "claim_type": ("stipulative", "formal", "empirical", "imperative"),
    "core_or_belt": ("core", "belt", "undeclared"),
    "imperative_form": ("categorical", "conditional", "not-applicable"),
    "integrity_status": (
        "clean", "corrected", "expression-of-concern", "retracted",
        "contested-unresolved", "not-checked",
    ),
    "confirmation_independence": ("same-lab", "same-lineage", "independent", "adversarial"),
    "replication_attempt_status": ("none-attempted", "succeeded", "failed", "mixed"),
}

#: Reliability floors from III.6, carried as declared conventions.
ALPHA_FLOOR = 0.67
ALPHA_FIRM = 0.80


class CodingError(ValueError):
    """A value outside its field's vocabulary."""


@dataclass(frozen=True, slots=True)
class CodedClaim:
    """One coder's judgement about one paper, in one graph."""

    graph_query: str
    paper_id: str
    coder: str
    coded_at: str
    values: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def value(self, name: str) -> str:
        return self.values.get(name, UNSET)


def current_coder() -> str:
    return os.environ.get(CODER_ENV, "").strip() or "anonymous"


def validate(values: Mapping[str, str]) -> dict[str, str]:
    """Reject anything outside the vocabularies; unknown fields are an error too."""
    clean: dict[str, str] = {}
    for name, value in values.items():
        if name not in FIELDS:
            raise CodingError(f"unknown coding field: {name!r}")
        if value != UNSET and value not in FIELDS[name]:
            allowed = ", ".join((*FIELDS[name], UNSET))
            raise CodingError(f"{name}={value!r} is not one of: {allowed}")
        clean[name] = value
    return clean


def prefill(node: Mapping[str, object]) -> dict[str, str]:
    """Defaults derivable from what the graph already holds — and nothing more.

    Only ``integrity_status`` is derivable, and only in one direction: OpenAlex's
    ``is_retracted`` establishes a retraction but its absence establishes nothing,
    so the other branch is ``not-checked`` rather than ``clean``.
    """
    return {
        "integrity_status": "retracted" if node.get("is_retracted") else "not-checked",
        **({"claim_type": "empirical"} if node.get("is_review") is False else {}),
    }


def _root(coding_dir: Path | None) -> Path:
    return coding_dir or global_store_path(CODING_DIRNAME)


def _path(graph_query: str, coding_dir: Path | None) -> Path:
    return _root(coding_dir) / f"{query_hash(graph_query)}.jsonl"


def record(
    graph_query: str,
    paper_id: str,
    values: Mapping[str, str],
    *,
    coder: str | None = None,
    note: str = "",
    coding_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> CodedClaim:
    """Append one coding decision. Never overwrites: disagreement is the data."""
    claim = CodedClaim(
        graph_query=graph_query,
        paper_id=paper_id,
        coder=coder or current_coder(),
        coded_at=(now or dt.datetime.now(dt.UTC)).isoformat(),
        values=validate(values),
        note=note,
    )
    path = _path(graph_query, coding_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(claim)) + "\n")
    return claim


def load(graph_query: str, *, coding_dir: Path | None = None) -> tuple[CodedClaim, ...]:
    """Every recorded judgement for a graph, oldest first. Corrupt lines are skipped."""
    path = _path(graph_query, coding_dir)
    if not path.exists():
        return ()
    claims: list[CodedClaim] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("paper_id"):
            claims.append(
                CodedClaim(
                    graph_query=str(data.get("graph_query", graph_query)),
                    paper_id=str(data["paper_id"]),
                    coder=str(data.get("coder", "anonymous")),
                    coded_at=str(data.get("coded_at", "")),
                    values={
                        k: str(v) for k, v in (data.get("values") or {}).items()
                    },
                    note=str(data.get("note", "")),
                )
            )
    return tuple(claims)


def latest_by_coder(claims: Sequence[CodedClaim], field_name: str) -> dict[str, dict[str, str]]:
    """``{paper_id: {coder: value}}`` using each coder's most recent judgement.

    A coder who revises keeps one vote; the superseded record stays on disk for
    audit but must not count twice toward agreement.
    """
    latest: dict[tuple[str, str], CodedClaim] = {}
    for claim in claims:
        if claim.value(field_name) == UNSET:
            continue
        key = (claim.paper_id, claim.coder)
        current = latest.get(key)
        if current is None or claim.coded_at >= current.coded_at:
            latest[key] = claim
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for (paper_id, coder), claim in latest.items():
        out[paper_id][coder] = claim.value(field_name)
    return dict(out)


def krippendorff_alpha(ratings: Mapping[str, Mapping[str, str]]) -> float | None:
    """Nominal Krippendorff's alpha, or ``None`` when it is undefined.

    Undefined rather than zero when fewer than two units carry two or more
    ratings, or when every rating is identical: with no variance there is no
    expected disagreement to divide by, and returning a number there would invent
    a reliability that was never measured.
    """
    units = [list(v.values()) for v in ratings.values() if len(v) >= 2]
    if len(units) < 2:
        return None

    coincidences: Counter[tuple[str, str]] = Counter()
    for values in units:
        pairs = len(values) - 1
        for i, first in enumerate(values):
            for j, second in enumerate(values):
                if i != j:
                    coincidences[(first, second)] += 1.0 / pairs  # type: ignore[assignment]

    totals: Counter[str] = Counter()
    for (first, _second), weight in coincidences.items():
        totals[first] += weight
    n = sum(totals.values())
    if n <= 1:
        return None

    observed = sum(w for (a, b), w in coincidences.items() if a != b)
    expected = sum(
        totals[a] * totals[b] for a in totals for b in totals if a != b
    ) / (n - 1)
    if expected <= 0:
        return None
    return 1.0 - (observed / expected)


@dataclass(frozen=True, slots=True)
class FieldReliability:
    field_name: str
    alpha: float | None
    units: int  # papers with two or more independent judgements
    coders: int

    @property
    def verdict(self) -> str:
        if self.alpha is None:
            return "not enough double-coded papers"
        if self.alpha >= ALPHA_FIRM:
            return "firm"
        if self.alpha >= ALPHA_FLOOR:
            return "usable"
        return "below floor — a finding about the framework, not a coding error"


def reliability(
    claims: Sequence[CodedClaim], field_name: str
) -> FieldReliability:
    """Inter-coder agreement for one field (III.6)."""
    ratings = latest_by_coder(claims, field_name)
    coders = {coder for votes in ratings.values() for coder in votes}
    return FieldReliability(
        field_name=field_name,
        alpha=krippendorff_alpha(ratings),
        units=sum(1 for votes in ratings.values() if len(votes) >= 2),
        coders=len(coders),
    )


def reliability_report(claims: Sequence[CodedClaim]) -> tuple[FieldReliability, ...]:
    """Agreement across every field that anyone has actually coded."""
    return tuple(
        reliability(claims, name)
        for name in FIELDS
        if any(claim.value(name) != UNSET for claim in claims)
    )


def coverage(claims: Sequence[CodedClaim], field_name: str) -> int:
    """Distinct papers with at least one judgement on this field."""
    return len({c.paper_id for c in claims if c.value(field_name) != UNSET})


def to_csv(claims: Iterable[CodedClaim]) -> str:
    """Flat table, one row per recorded judgement, every field a column."""
    import csv

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["paper_id", "coder", "coded_at", *FIELDS, "note"])
    for claim in claims:
        writer.writerow(
            [claim.paper_id, claim.coder, claim.coded_at]
            + [claim.value(name) for name in FIELDS]
            + [claim.note]
        )
    return buffer.getvalue()
