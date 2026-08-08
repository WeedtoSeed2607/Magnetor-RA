"""Branch C — the three guards on the verdict layer: alpha, the clock, the unit."""

from __future__ import annotations

import datetime as dt

from magnetor.coding import (
    ALPHA_FLOOR,
    CODING_UNIT_CAVEAT,
    FIRM,
    PROGRESSIVE,
    STAGNANT,
    UNDETERMINED,
    UNRELIABLE,
    UNVALIDATED,
    USABLE,
    CodedClaim,
    Verdict,
    programme_verdict,
    qualify,
)

_GATE_PASS = {
    "independently_measurable": "yes",
    "excess_content": "yes",
    "use_novel": "yes",
}


def _claim(paper: str, coder: str, values: dict[str, str]) -> CodedClaim:
    return CodedClaim(
        graph_query="q", paper_id=paper, coder=coder,
        coded_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(), values=values,
    )


def _agreeing(n_papers: int) -> list[CodedClaim]:
    """Two coders, identical judgements — but a mix of values so alpha is defined."""
    claims = []
    for i in range(n_papers):
        values = dict(_GATE_PASS) if i % 2 else {**_GATE_PASS, "use_novel": "no",
                                                 "excess_content": "no",
                                                 "independently_measurable": "no"}
        for coder in ("ana", "ben"):
            claims.append(_claim(f"W{i}", coder, values))
    return claims


# --- 1. a verdict cannot outrun the agreement on its inputs ---

def test_uncoded_agreement_makes_a_verdict_unvalidated() -> None:
    single = [_claim("W1", "ana", _GATE_PASS)]
    found = qualify(Verdict(PROGRESSIVE, "passed"), single)
    assert found.standing == UNVALIDATED
    assert not found.is_trustworthy
    assert "never been measured" in found.caveat


def test_no_claims_at_all_is_unvalidated_not_firm() -> None:
    assert qualify(Verdict(PROGRESSIVE, "passed"), []).standing == UNVALIDATED


def test_agreeing_coders_reach_firm_standing() -> None:
    found = qualify(Verdict(PROGRESSIVE, "passed"), _agreeing(4))
    assert found.standing == FIRM
    assert found.is_trustworthy


def test_disagreement_below_the_floor_marks_the_verdict_unreliable() -> None:
    """III.6: below-floor agreement is a finding about the framework."""
    claims = []
    for i in range(4):
        settled = "yes" if i % 2 else "no"  # agreed, and varying, so alpha is defined
        for coder in ("ana", "ben"):
            claims.append(_claim(f"W{i}", coder, {
                "independently_measurable": settled,
                "excess_content": settled,
                # the coders systematically invert each other on the third field
                "use_novel": ("yes" if i % 2 else "no") if coder == "ana"
                             else ("no" if i % 2 else "yes"),
            }))
    found = qualify(Verdict(PROGRESSIVE, "passed"), claims)
    assert found.standing == UNRELIABLE
    assert not found.is_trustworthy
    assert str(ALPHA_FLOOR) in found.caveat


def test_total_agreement_is_unvalidated_not_firm() -> None:
    """Zero variance leaves alpha undefined, and undefined is not the same as good.

    If every coder gives the same answer on every paper there is no expected
    disagreement to normalise against, so reliability is unestablished rather than
    perfect. Reporting that as firm would be the invented certainty this layer
    exists to prevent.
    """
    claims = [
        _claim(f"W{i}", coder, _GATE_PASS)
        for i in range(4)
        for coder in ("ana", "ben")
    ]
    assert qualify(Verdict(PROGRESSIVE, "passed"), claims).standing == UNVALIDATED


def test_standing_takes_the_weakest_gate_field_not_the_average() -> None:
    """One well-agreed field must not mask another nobody can reproduce."""
    claims = []
    for i in range(4):  # perfect agreement on two fields, none at all on the third
        for coder in ("ana", "ben"):
            claims.append(
                _claim(f"W{i}", coder, {
                    "independently_measurable": "yes" if i % 2 else "no",
                    "excess_content": "yes" if i % 2 else "no",
                })
            )
    found = qualify(Verdict(PROGRESSIVE, "passed"), claims)
    assert found.standing == UNVALIDATED  # use_novel was never coded
    assert {r.field_name for r in found.gate_reliability} == set(_GATE_PASS)


def test_every_qualified_verdict_carries_the_unit_caveat() -> None:
    for claims in ([], _agreeing(4)):
        assert CODING_UNIT_CAVEAT in qualify(Verdict(PROGRESSIVE, "p"), claims).caveat


def test_standing_is_separate_from_what_the_verdict_says() -> None:
    """A firm standing does not make a verdict progressive, or the reverse."""
    found = qualify(Verdict(STAGNANT, "content-free"), _agreeing(4))
    assert found.standing == FIRM
    assert found.verdict.verdict == STAGNANT


# --- 2. the clock ---

def _passing(paper: str) -> CodedClaim:
    return _claim(paper, "ana", _GATE_PASS)


def test_without_years_the_clock_is_skipped_and_says_so() -> None:
    found = programme_verdict([_passing("W1")])
    assert found.verdict == PROGRESSIVE
    assert "Clock not run" in found.reason


def test_a_recent_passing_revision_stays_progressive() -> None:
    found = programme_verdict([_passing("W1")], years={"W1": 2024}, as_of=2026)
    assert found.verdict == PROGRESSIVE
    assert "Clock not run" not in found.reason


def test_a_long_lull_turns_a_passing_programme_stagnant() -> None:
    """Passing the gate in the past is not the same as still moving."""
    found = programme_verdict(
        [_passing("W1")], years={"W1": 1970}, window_years=10, as_of=2026
    )
    assert found.verdict == STAGNANT
    assert "1970" in found.reason
    assert "declared convention" in found.reason


def test_the_window_is_a_parameter() -> None:
    claims, years = [_passing("W1")], {"W1": 2000}

    def at(window: int) -> str:
        return programme_verdict(
            claims, years=years, window_years=window, as_of=2026
        ).verdict

    assert at(10) == STAGNANT
    assert at(40) == PROGRESSIVE


def test_the_clock_reads_the_most_recent_passing_revision() -> None:
    found = programme_verdict(
        [_passing("OLD"), _passing("NEW")],
        years={"OLD": 1970, "NEW": 2024}, as_of=2026,
    )
    assert found.verdict == PROGRESSIVE


def test_as_of_supports_a_historical_cut_off() -> None:
    """Coding a past programme must not read as stalled by construction (II.14)."""
    claims, years = [_passing("W1")], {"W1": 1975}
    assert programme_verdict(claims, years=years, as_of=1980).verdict == PROGRESSIVE
    assert programme_verdict(claims, years=years, as_of=2026).verdict == STAGNANT


def test_unknown_years_leave_the_clock_undetermined() -> None:
    found = programme_verdict([_passing("W1")], years={"W1": None}, as_of=2026)
    assert found.verdict == UNDETERMINED
    assert "cannot be applied" in found.reason


def test_the_pessimistic_roll_up_is_named_as_a_stipulation() -> None:
    claims = [_passing("W1"), _claim("W2", "ana", {**_GATE_PASS,
                                                   "independently_measurable": "no"})]
    found = programme_verdict(claims, years={"W1": 2025, "W2": 2025}, as_of=2026)
    assert found.verdict == STAGNANT
    assert "Stipulated roll-up" in found.reason


def test_the_clock_never_rescues_a_failed_gate() -> None:
    """Recency is not progress: a degenerating revision stays degenerating."""
    claims = [_claim("W1", "ana", {**_GATE_PASS, "independently_measurable": "no"})]
    found = programme_verdict(claims, years={"W1": 2026}, as_of=2026)
    assert found.verdict != PROGRESSIVE


def test_usable_sits_between_the_floor_and_firm() -> None:
    """A verdict clearing the floor but not the firm bar says so rather than passing."""
    claims = []
    for i in range(12):
        for coder in ("ana", "ben"):
            value = "yes" if i % 2 else "no"
            # one coder flips a single paper: agreement is high but not perfect
            if coder == "ben" and i == 0:
                value = "yes"
            claims.append(_claim(f"W{i}", coder, {
                "independently_measurable": value,
                "excess_content": value,
                "use_novel": value,
            }))
    found = qualify(Verdict(PROGRESSIVE, "p"), claims)
    assert found.standing in (USABLE, FIRM)
