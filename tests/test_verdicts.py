"""Branch C — the Lakatosian gate (II.9) and programme-level verdicts."""

from __future__ import annotations

import datetime as dt

from magnetor.coding import (
    DEGENERATING,
    PROGRESSIVE,
    STAGNANT,
    UNDETERMINED,
    UNSET,
    CodedClaim,
    judge,
    programme_verdict,
)

_PASSES = {
    "independently_measurable": "yes",
    "excess_content": "yes",
    "use_novel": "yes",
}


def _claim(values: dict[str, str], paper: str = "W1") -> CodedClaim:
    return CodedClaim(
        graph_query="q", paper_id=paper, coder="ana",
        coded_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(), values=values,
    )


def test_all_three_criteria_pass_is_progressive() -> None:
    found = judge(_PASSES)
    assert found.verdict == PROGRESSIVE
    assert "use-novel" in found.reason


def test_unmeasurable_addition_is_degenerating() -> None:
    """Criterion 1 is the ad hoc rescue test: a term measurable only by its own fit."""
    found = judge({**_PASSES, "independently_measurable": "no"})
    assert found.verdict == DEGENERATING


def test_measurable_but_content_free_is_stagnant() -> None:
    """The category a progressive/degenerating binary would misreport."""
    assert judge({**_PASSES, "excess_content": "no"}).verdict == STAGNANT
    assert judge({**_PASSES, "use_novel": "no"}).verdict == STAGNANT


def test_uncoded_gate_is_undetermined_never_progressive() -> None:
    """Absence of falsification is not corroboration (III.4E)."""
    found = judge({"independently_measurable": "yes"})
    assert found.verdict == UNDETERMINED
    assert "excess_content" in found.reason
    assert judge({}).verdict == UNDETERMINED
    assert judge(dict.fromkeys(_PASSES, UNSET)).verdict == UNDETERMINED


def test_undeterminable_use_novelty_does_not_pass_the_gate() -> None:
    found = judge({**_PASSES, "use_novel": "undeterminable"})
    assert found.verdict == UNDETERMINED
    assert "requires reading" in found.reason


def test_strength_grades_never_decide_a_verdict() -> None:
    """Criteria 4 and 5 are grades in the source, so they cannot buy off a failure."""
    weak = judge({**_PASSES, "parsimony_out_of_sample": "worse",
                  "form_derived_independently": "no"})
    assert weak.verdict == PROGRESSIVE
    strong = judge({**_PASSES, "independently_measurable": "no",
                    "parsimony_out_of_sample": "improved",
                    "form_derived_independently": "yes"})
    assert strong.verdict == DEGENERATING


def test_retracted_core_evidence_defeats_a_verdict() -> None:
    found = judge({**_PASSES, "integrity_status": "retracted", "core_or_belt": "core"})
    assert found.verdict == PROGRESSIVE  # the gate still passed...
    assert found.is_defeated  # ...but the verdict is defeated, not downgraded
    assert "retracted" in found.defeated_by


def test_peripheral_retraction_does_not_defeat() -> None:
    """Otherwise every large programme acquires a flag by base rate alone."""
    found = judge({**_PASSES, "integrity_status": "retracted", "core_or_belt": "belt"})
    assert not found.is_defeated


def test_same_lab_confirmation_and_untried_replication_defeat() -> None:
    assert judge({**_PASSES, "confirmation_independence": "same-lab"}).is_defeated
    assert judge({**_PASSES, "replication_attempt_status": "none-attempted"}).is_defeated
    assert not judge({**_PASSES, "confirmation_independence": "adversarial"}).is_defeated


def test_programme_verdict_is_undetermined_without_coded_revisions() -> None:
    assert programme_verdict([]).verdict == UNDETERMINED
    assert programme_verdict([_claim({})]).verdict == UNDETERMINED


def test_one_degenerating_revision_holds_the_programme_back() -> None:
    claims = [
        _claim(_PASSES, "W1"),
        _claim(_PASSES, "W2"),
        _claim({**_PASSES, "independently_measurable": "no"}, "W3"),
    ]
    found = programme_verdict(claims)
    assert found.verdict == STAGNANT
    assert "not progressive because some of its revisions are" in found.reason


def test_all_progressive_rolls_up_to_progressive() -> None:
    claims = [_claim(_PASSES, "W1"), _claim(_PASSES, "W2")]
    assert programme_verdict(claims).verdict == PROGRESSIVE


def test_no_passing_revision_rolls_up_to_the_worst_seen() -> None:
    assert programme_verdict(
        [_claim({**_PASSES, "independently_measurable": "no"})]
    ).verdict == DEGENERATING
    assert programme_verdict(
        [_claim({**_PASSES, "excess_content": "no"})]
    ).verdict == STAGNANT
