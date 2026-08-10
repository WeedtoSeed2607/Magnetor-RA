"""The failsafe: a cause is never reported without the evidence that establishes it."""

from __future__ import annotations

import pytest

from magnetor.diagnostics import (
    NO_API_KEY,
    NO_CLOSE_MATCH,
    NOTHING_COLLECTED,
    SUBJECT_EMPTY,
    TRENDS_NOT_RUN,
    TRENDS_TOO_LITTLE_DATA,
    UNKNOWN,
    WRONG_LOCATION,
    Diagnosis,
    diagnose_no_results,
    diagnose_no_trends,
)

_STOCKED = {"qm": 130, "math": 130, "neuro": 134, "philosophy": 0}


# --- the failsafe itself -------------------------------------------------

def _all_diagnoses() -> list[Diagnosis]:
    """Every diagnosis both functions can produce, including the unknowns."""
    return [
        diagnose_no_results(api_key_present=None, stocked=None, forced_subject=None),
        diagnose_no_results(api_key_present=False, stocked=_STOCKED, forced_subject=None),
        diagnose_no_results(api_key_present=True, stocked=None, forced_subject=None),
        diagnose_no_results(api_key_present=True, stocked={"qm": 0}, forced_subject=None),
        diagnose_no_results(
            api_key_present=True, stocked={"qm": 0}, forced_subject=None,
            elsewhere={"/elsewhere": 12},
        ),
        diagnose_no_results(
            api_key_present=True, stocked=_STOCKED, forced_subject="philosophy"
        ),
        diagnose_no_results(api_key_present=True, stocked=_STOCKED, forced_subject="qm"),
        diagnose_no_trends(records=None, trends_present=None),
        diagnose_no_trends(records=0, trends_present=False),
        diagnose_no_trends(records=130, trends_present=None),
        diagnose_no_trends(records=130, trends_present=False),
        diagnose_no_trends(records=130, trends_present=True),
    ]


def test_an_identified_cause_always_carries_evidence_that_was_checked() -> None:
    """The core guarantee. A cause with no conclusive observation behind it is a guess."""
    for found in _all_diagnoses():
        if found.is_identified:
            assert found.supporting(), found.cause


def test_an_unknown_cause_never_names_a_culprit() -> None:
    """When something could not be checked, the message must not blame anything."""
    blamed = ("embed", "api key", "collect", "acquire", "trends")
    for found in _all_diagnoses():
        if not found.is_identified:
            text = (found.headline + " " + found.remedy).lower()
            assert "could not be determined" in text
            assert not any(word in found.headline.lower() for word in blamed)


def test_every_diagnosis_offers_something_to_do() -> None:
    for found in _all_diagnoses():
        assert len(found.remedy) > 20, found.cause


def test_an_unmade_observation_is_distinguishable_from_a_false_one() -> None:
    """`None` means "could not look", `False` means "looked, and no"."""
    unknown = diagnose_no_results(api_key_present=None, stocked=_STOCKED, forced_subject=None)
    missing = diagnose_no_results(api_key_present=False, stocked=_STOCKED, forced_subject=None)
    assert unknown.cause == UNKNOWN
    assert missing.cause == NO_API_KEY
    assert not unknown.evidence[0].conclusive
    assert missing.evidence[0].conclusive


# --- search diagnoses ----------------------------------------------------

def test_a_missing_key_is_not_blamed_on_the_corpus() -> None:
    """The exact bug this exists to prevent."""
    found = diagnose_no_results(
        api_key_present=False, stocked=_STOCKED, forced_subject=None
    )
    assert found.cause == NO_API_KEY
    assert "embed" not in found.headline.lower()
    assert "nothing was wrong with your papers" in found.remedy.lower()


def test_an_empty_corpus_is_reported_as_such() -> None:
    found = diagnose_no_results(
        api_key_present=True, stocked={"qm": 0, "math": 0}, forced_subject=None
    )
    assert found.cause == NOTHING_COLLECTED


def test_papers_found_elsewhere_is_a_wrong_folder_not_an_empty_corpus() -> None:
    """An empty folder and a misdirected app give the same count.

    Only one of them is fixed by collecting papers, so telling a user with a full
    corpus to `acquire all` wastes their time on work already done.
    """
    found = diagnose_no_results(
        api_key_present=True,
        stocked={"qm": 0, "math": 0},
        forced_subject=None,
        location=r"C:\wrong\place",
        elsewhere={r"C:\right\place": 394, r"C:\empty": 0},
    )
    assert found.cause == WRONG_LOCATION
    assert r"C:\right\place" in found.remedy
    assert "collecting" in found.remedy  # explicitly says not to re-collect
    assert "acquire all" not in found.remedy
    assert r"C:\empty" not in found.evidence[-1].detail  # only real finds listed


def test_nothing_anywhere_is_still_an_empty_corpus() -> None:
    found = diagnose_no_results(
        api_key_present=True,
        stocked={"qm": 0},
        forced_subject=None,
        location="/here",
        elsewhere={"/there": 0},
    )
    assert found.cause == NOTHING_COLLECTED
    assert found.evidence[-1].observed is False


def test_alternatives_are_only_consulted_when_the_active_root_is_empty() -> None:
    found = diagnose_no_results(
        api_key_present=True,
        stocked=_STOCKED,
        forced_subject=None,
        elsewhere={"/somewhere": 999},
    )
    assert found.cause == NO_CLOSE_MATCH  # a stocked corpus is not misdirected


def test_an_empty_count_always_says_where_it_looked() -> None:
    """Zero has two causes — nothing collected, or the wrong folder read.

    Without the location the message cannot separate them, which is this module's
    own failure mode reappearing one level down.
    """
    found = diagnose_no_results(
        api_key_present=True,
        stocked={"qm": 0},
        forced_subject=None,
        location=r"C:\some\where",
    )
    assert found.cause == NOTHING_COLLECTED
    assert r"C:\some\where" in found.evidence[-1].detail
    assert "wrong folder" in found.remedy


def test_a_non_empty_count_also_reports_the_location() -> None:
    found = diagnose_no_results(
        api_key_present=True, stocked=_STOCKED, forced_subject="qm", location="/data"
    )
    assert "/data" in found.evidence[1].detail


def test_searching_only_an_empty_subject_names_the_stocked_ones() -> None:
    found = diagnose_no_results(
        api_key_present=True, stocked=_STOCKED, forced_subject="philosophy"
    )
    assert found.cause == SUBJECT_EMPTY
    assert "philosophy" in found.headline
    for name in ("qm", "math", "neuro"):
        assert name in found.remedy


def test_a_stocked_subject_that_simply_missed_is_not_called_empty() -> None:
    found = diagnose_no_results(
        api_key_present=True, stocked=_STOCKED, forced_subject="qm"
    )
    assert found.cause == NO_CLOSE_MATCH
    assert "qm (130)" in found.remedy


def test_the_key_is_checked_before_the_corpus() -> None:
    """A search that never ran says nothing about the papers."""
    found = diagnose_no_results(
        api_key_present=False, stocked={"qm": 0}, forced_subject="qm"
    )
    assert found.cause == NO_API_KEY  # not NOTHING_COLLECTED
    assert len(found.evidence) == 1  # and it stopped looking


# --- trend diagnoses -----------------------------------------------------

def test_an_empty_subject_is_not_told_to_run_trends() -> None:
    """Running the analysis on zero papers changes nothing, so do not suggest it."""
    found = diagnose_no_trends(records=0, trends_present=False)
    assert found.cause == NOTHING_COLLECTED
    assert "magnetor trends" not in found.remedy


def test_a_stocked_subject_with_no_analysis_is_told_to_run_it() -> None:
    found = diagnose_no_trends(records=130, trends_present=False)
    assert found.cause == TRENDS_NOT_RUN
    assert "magnetor trends" in found.remedy


def test_analysis_that_ran_but_found_nothing_is_named_separately() -> None:
    found = diagnose_no_trends(records=130, trends_present=True)
    assert found.cause == TRENDS_TOO_LITTLE_DATA
    assert "volume gate" in found.remedy


@pytest.mark.parametrize(
    ("records", "present"), [(None, None), (None, True), (130, None)]
)
def test_trends_cannot_be_diagnosed_from_incomplete_facts(records, present) -> None:
    assert diagnose_no_trends(records=records, trends_present=present).cause == UNKNOWN
