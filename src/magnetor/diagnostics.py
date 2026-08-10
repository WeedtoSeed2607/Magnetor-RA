"""Why did that come back empty? — answered only from things actually checked.

An empty result has several possible causes and they need different fixes. The
dashboard used to name one of them for every case: *"has this domain been
embedded?"*, which sent the reader to ``magnetor embed`` when the real cause was
a missing API key, an empty subject, or simply nothing close enough. A message
that points at the wrong problem costs more than no message, because the reader
spends their time where the fault is not.

**The failsafe is the shape of this module, not extra branches.** A cause is only
ever returned alongside the observations that establish it, and any observation
that could not be made yields :data:`UNKNOWN` — never a guess that happens to
sound plausible. Adding a new cause therefore forces you to add the check that
justifies it, and the tests refuse a diagnosis whose evidence does not support it.

The functions are pure: callers gather the facts (which is where things can fail)
and pass them in, so every branch is reachable in a test without a corpus, an API
key or a network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Causes. UNKNOWN is not a failure mode of last resort — it is the correct answer
# whenever the evidence is incomplete, and it is what stops this becoming a
# guessing machine again.
UNKNOWN = "unknown"
NO_API_KEY = "no-api-key"
NOTHING_COLLECTED = "nothing-collected"
SUBJECT_EMPTY = "subject-empty"
NO_CLOSE_MATCH = "no-close-match"
TRENDS_NOT_RUN = "trends-not-run"
TRENDS_TOO_LITTLE_DATA = "trends-too-little-data"


@dataclass(frozen=True, slots=True)
class Check:
    """One thing that was looked at, and what was seen.

    ``observed`` is ``None`` when the check could not be carried out at all —
    the distinction between "we looked and it was false" and "we could not look"
    is exactly what keeps a diagnosis honest.
    """

    name: str
    observed: bool | None
    detail: str

    @property
    def conclusive(self) -> bool:
        return self.observed is not None


@dataclass(frozen=True, slots=True)
class Diagnosis:
    cause: str
    headline: str  # what went wrong, in the reader's terms
    remedy: str  # what to do about it
    evidence: tuple[Check, ...]  # everything looked at, in order

    @property
    def is_identified(self) -> bool:
        return self.cause != UNKNOWN

    def supporting(self) -> tuple[Check, ...]:
        """Checks that were actually carried out."""
        return tuple(c for c in self.evidence if c.conclusive)


_UNKNOWN_HEADLINE = "Nothing came back, and the reason could not be determined."
_UNKNOWN_REMEDY = (
    "The checks below could not all be carried out, so no cause is being claimed. "
    "Run `magnetor status all` to see the corpus directly."
)


def _unknown(evidence: tuple[Check, ...]) -> Diagnosis:
    return Diagnosis(UNKNOWN, _UNKNOWN_HEADLINE, _UNKNOWN_REMEDY, evidence)


def diagnose_no_results(
    *,
    api_key_present: bool | None,
    stocked: Mapping[str, int] | None,
    forced_subject: str | None,
) -> Diagnosis:
    """Why a search returned nothing.

    Ordered by how fundamental the fault is: a missing key means the search never
    ran at all, so reporting anything about the corpus would be beside the point.
    """
    evidence: list[Check] = [
        Check(
            "search key configured",
            api_key_present,
            "found" if api_key_present else "not set" if api_key_present is not None
            else "could not be checked",
        )
    ]
    if api_key_present is None:
        return _unknown(tuple(evidence))
    if not api_key_present:
        return Diagnosis(
            NO_API_KEY,
            "The search could not run — no key is configured for the service that "
            "understands your question.",
            "Put `MAGNETOR_VOYAGE_API_KEY=...` in the project's `.env`, then restart. "
            "Nothing was wrong with your papers.",
            tuple(evidence),
        )

    if stocked is None:
        evidence.append(Check("papers collected", None, "could not be counted"))
        return _unknown(tuple(evidence))

    total = sum(stocked.values())
    stocked_names = [name for name, n in stocked.items() if n > 0]
    evidence.append(
        Check(
            "papers collected",
            total > 0,
            f"{total} across {len(stocked_names)} subject(s)"
            if total
            else "none in any subject",
        )
    )
    if total == 0:
        return Diagnosis(
            NOTHING_COLLECTED,
            "There are no papers to search yet.",
            "Collect some first: `magnetor acquire all`, then `magnetor embed all`.",
            tuple(evidence),
        )

    if forced_subject is not None:
        held = stocked.get(forced_subject, 0)
        evidence.append(
            Check(
                f"papers in {forced_subject}",
                held > 0,
                f"{held} paper(s)" if held else "none",
            )
        )
        if held == 0:
            return Diagnosis(
                SUBJECT_EMPTY,
                f"The search was limited to {forced_subject}, which has no papers yet.",
                "Untick the limit to search everything, or pick a subject that has "
                f"papers: {', '.join(stocked_names)}.",
                tuple(evidence),
            )

    # Everything needed was present and the search still matched nothing. That is
    # an observation, not an inference — the search did run, and returned zero.
    return Diagnosis(
        NO_CLOSE_MATCH,
        "The search ran, but nothing came close enough to your wording.",
        "The collection covers "
        + ", ".join(f"{n} ({stocked[n]})" for n in stocked_names)
        + ". Try wording nearer one of those, or collect the subject you want.",
        tuple(evidence),
    )


def diagnose_no_trends(
    *,
    records: int | None,
    trends_present: bool | None,
) -> Diagnosis:
    """Why a subject shows no trend lines.

    The old message told the reader to run ``magnetor trends`` in every case,
    including when the subject held no papers at all — where running it changes
    nothing, because there is nothing to analyse.
    """
    evidence: list[Check] = [
        Check(
            "papers collected",
            None if records is None else records > 0,
            "could not be counted" if records is None else f"{records} paper(s)",
        )
    ]
    if records is None:
        return _unknown(tuple(evidence))
    if records == 0:
        return Diagnosis(
            NOTHING_COLLECTED,
            "This subject has no papers, so there is nothing to find trends in.",
            "Collect papers first. For subjects without an automatic feed, drop "
            "records into the subject's inbox folder, then run `magnetor acquire`.",
            tuple(evidence),
        )

    evidence.append(
        Check(
            "trend analysis run",
            trends_present,
            "results found" if trends_present else "never run"
            if trends_present is not None else "could not be checked",
        )
    )
    if trends_present is None:
        return _unknown(tuple(evidence))
    if not trends_present:
        return Diagnosis(
            TRENDS_NOT_RUN,
            "The papers are here, but the trend analysis has not been run on them.",
            "Run `magnetor trends <subject>`.",
            tuple(evidence),
        )
    return Diagnosis(
        TRENDS_TOO_LITTLE_DATA,
        "The analysis ran but found too little recent activity to report on.",
        "This is the volume gate doing its job — a handful of papers cannot "
        "support a trend. Collect more, or widen the time slice.",
        tuple(evidence),
    )
