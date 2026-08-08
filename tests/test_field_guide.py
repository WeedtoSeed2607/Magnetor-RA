"""Branch C — the coder-facing guide: every field explained, every role assigned."""

from __future__ import annotations

from magnetor.coding import (
    _GATE,
    CONTEXT,
    DEFEATER,
    FIELD_GUIDE,
    FIELDS,
    GATE,
    fields_by_role,
)


def test_every_guided_field_exists_in_the_schema() -> None:
    assert set(FIELD_GUIDE) <= set(FIELDS)


def test_the_gate_role_matches_the_fields_the_gate_actually_reads() -> None:
    """If these drift apart the form teaches the wrong thing about what decides."""
    assert set(fields_by_role(GATE)) == set(_GATE)


def test_every_value_in_a_guided_vocabulary_is_explained() -> None:
    """A coder must never meet an option the guide cannot define."""
    for name, guide in FIELD_GUIDE.items():
        assert set(guide.values) == set(FIELDS[name]), name


def test_every_guide_states_a_question_a_place_to_look_and_an_effect() -> None:
    for name, guide in FIELD_GUIDE.items():
        assert guide.question.endswith("?"), name
        assert len(guide.look_for) > 40, name
        assert len(guide.effect) > 20, name


def test_roles_partition_the_guided_fields() -> None:
    roles = [GATE, DEFEATER, CONTEXT]
    covered = [name for role in roles for name in fields_by_role(role)]
    assert sorted(covered) == sorted(FIELD_GUIDE)
    assert len(covered) == len(set(covered))  # no field in two roles


def test_each_role_is_populated() -> None:
    assert len(fields_by_role(GATE)) == 3
    assert fields_by_role(DEFEATER)
    assert fields_by_role(CONTEXT)


def test_gate_guides_name_their_failure_outcome() -> None:
    """The point of the grouping: a coder should see what an answer costs."""
    outcomes = {
        "independently_measurable": "degenerating",
        "excess_content": "stagnant",
        "use_novel": "stagnant",
    }
    for name, outcome in outcomes.items():
        assert outcome in FIELD_GUIDE[name].effect.lower(), name


def test_defeater_guides_say_they_only_void() -> None:
    for name in fields_by_role(DEFEATER):
        effect = FIELD_GUIDE[name].effect.lower()
        assert "defeat" in effect or "decides whether" in effect, name


def test_not_checked_is_never_described_as_clean() -> None:
    """The prefill's whole point: silence from one source establishes nothing."""
    meaning = FIELD_GUIDE["integrity_status"].values["not-checked"].lower()
    assert "never assume" in meaning
