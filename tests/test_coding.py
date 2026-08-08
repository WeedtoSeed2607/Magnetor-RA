"""Branch C — the Lakatosian coding instrument and its reliability statistic."""

from __future__ import annotations

import datetime as dt

import pytest

from magnetor.coding import (
    ALPHA_FLOOR,
    UNSET,
    CodedClaim,
    CodingError,
    krippendorff_alpha,
    latest_by_coder,
    load,
    prefill,
    record,
    reliability,
    reliability_report,
    to_csv,
    validate,
)

_QUERY = "Defining Morality"


def _at(day: int) -> dt.datetime:
    return dt.datetime(2026, 1, day, tzinfo=dt.UTC)


def test_validate_accepts_the_vocabulary_and_unset() -> None:
    assert validate({"explanation_type": "normative"}) == {"explanation_type": "normative"}
    assert validate({"level_sense": UNSET}) == {"level_sense": UNSET}


def test_validate_rejects_values_and_fields_outside_the_manual() -> None:
    with pytest.raises(CodingError, match="not one of"):
        validate({"explanation_type": "vibes"})
    with pytest.raises(CodingError, match="unknown coding field"):
        validate({"made_up": "x"})


def test_prefill_never_asserts_clean_from_absence() -> None:
    """OpenAlex flags retractions; its silence establishes nothing."""
    assert prefill({"is_retracted": True})["integrity_status"] == "retracted"
    assert prefill({"is_retracted": False})["integrity_status"] == "not-checked"
    assert prefill({})["integrity_status"] == "not-checked"


def test_record_appends_and_load_round_trips(tmp_path) -> None:
    record(_QUERY, "W1", {"explanation_type": "normative"}, coder="ana", coding_dir=tmp_path)
    record(_QUERY, "W2", {"explanation_type": "descriptive"}, coder="ben", coding_dir=tmp_path)
    claims = load(_QUERY, coding_dir=tmp_path)
    assert [c.paper_id for c in claims] == ["W1", "W2"]
    assert claims[0].coder == "ana"


def test_recording_never_overwrites_a_disagreement(tmp_path) -> None:
    """Two coders differing is the measurement, not a conflict to resolve."""
    record(_QUERY, "W1", {"explanation_type": "normative"}, coder="ana", coding_dir=tmp_path)
    record(_QUERY, "W1", {"explanation_type": "descriptive"}, coder="ben", coding_dir=tmp_path)
    assert len(load(_QUERY, coding_dir=tmp_path)) == 2


def test_load_is_empty_for_an_uncoded_graph(tmp_path) -> None:
    assert load("never coded", coding_dir=tmp_path) == ()


def test_corrupt_lines_are_skipped(tmp_path) -> None:
    record(_QUERY, "W1", {"claim_type": "empirical"}, coder="ana", coding_dir=tmp_path)
    path = next(tmp_path.glob("*.jsonl"))
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    assert len(load(_QUERY, coding_dir=tmp_path)) == 1


def test_a_revision_replaces_the_coders_earlier_vote(tmp_path) -> None:
    record(_QUERY, "W1", {"core_or_belt": "core"}, coder="ana",
           coding_dir=tmp_path, now=_at(1))
    record(_QUERY, "W1", {"core_or_belt": "belt"}, coder="ana",
           coding_dir=tmp_path, now=_at(2))
    votes = latest_by_coder(load(_QUERY, coding_dir=tmp_path), "core_or_belt")
    assert votes["W1"] == {"ana": "belt"}  # one coder, one vote, the later one


def test_unset_values_do_not_count_as_judgements(tmp_path) -> None:
    record(_QUERY, "W1", {"level_sense": UNSET}, coder="ana", coding_dir=tmp_path)
    assert latest_by_coder(load(_QUERY, coding_dir=tmp_path), "level_sense") == {}


def _claims(pairs: dict[str, dict[str, str]]) -> list[CodedClaim]:
    return [
        CodedClaim(
            graph_query=_QUERY, paper_id=paper, coder=coder,
            coded_at=_at(1).isoformat(), values={"explanation_type": value},
        )
        for paper, votes in pairs.items()
        for coder, value in votes.items()
    ]


def test_perfect_agreement_scores_one() -> None:
    ratings = {"W1": {"a": "descriptive", "b": "descriptive"},
               "W2": {"a": "normative", "b": "normative"}}
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_systematic_disagreement_scores_at_or_below_zero() -> None:
    ratings = {"W1": {"a": "descriptive", "b": "normative"},
               "W2": {"a": "normative", "b": "descriptive"}}
    alpha = krippendorff_alpha(ratings)
    assert alpha is not None and alpha <= 0.0


def test_alpha_matches_a_hand_computed_value() -> None:
    """Four units, two coders, one disagreement — worked by hand from the definition.

    Coincidences: o[1,1]=4, o[2,2]=2, o[1,2]=o[2,1]=1, so n_1=5, n_2=3, n=8.
    Observed disagreement = 2; expected = (5*3 + 3*5)/(8-1) = 30/7.
    alpha = 1 - 2/(30/7) = 0.5333...
    """
    ratings = {
        "W1": {"a": "1", "b": "1"},
        "W2": {"a": "1", "b": "1"},
        "W3": {"a": "2", "b": "2"},
        "W4": {"a": "1", "b": "2"},
    }
    assert krippendorff_alpha(ratings) == pytest.approx(1 - 2 / (30 / 7))


def test_alpha_is_undefined_rather_than_zero_without_enough_data() -> None:
    """No variance means no expected disagreement to divide by."""
    assert krippendorff_alpha({}) is None
    assert krippendorff_alpha({"W1": {"a": "descriptive", "b": "descriptive"}}) is None
    # Two units, but every rating identical -> expected disagreement is zero.
    assert krippendorff_alpha(
        {"W1": {"a": "x", "b": "x"}, "W2": {"a": "x", "b": "x"}}
    ) is None


def test_singly_coded_papers_are_excluded() -> None:
    ratings = {"W1": {"a": "descriptive"}, "W2": {"a": "normative"}}
    assert krippendorff_alpha(ratings) is None


def test_reliability_reports_units_coders_and_a_verdict() -> None:
    claims = _claims({
        "W1": {"ana": "descriptive", "ben": "descriptive"},
        "W2": {"ana": "normative", "ben": "normative"},
        "W3": {"ana": "mechanistic", "ben": "descriptive"},
    })
    found = reliability(claims, "explanation_type")
    assert found.units == 3
    assert found.coders == 2
    assert found.alpha is not None


def test_a_field_below_the_floor_is_framed_as_a_finding() -> None:
    claims = _claims({
        "W1": {"ana": "descriptive", "ben": "normative"},
        "W2": {"ana": "normative", "ben": "descriptive"},
    })
    found = reliability(claims, "explanation_type")
    assert found.alpha is not None and found.alpha < ALPHA_FLOOR
    assert "finding about the framework" in found.verdict


def test_report_covers_only_fields_anyone_coded() -> None:
    claims = _claims({"W1": {"ana": "descriptive", "ben": "descriptive"},
                      "W2": {"ana": "normative", "ben": "normative"}})
    names = {r.field_name for r in reliability_report(claims)}
    assert names == {"explanation_type"}


def test_csv_has_a_row_per_judgement_and_a_column_per_field() -> None:
    claims = _claims({"W1": {"ana": "descriptive", "ben": "normative"}})
    text = to_csv(claims)
    assert text.count("\n") == 3  # header plus two rows
    assert "explanation_type" in text.splitlines()[0]
    assert "level_sense" in text.splitlines()[0]  # uncoded fields still get a column
