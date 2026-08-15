"""Stage 1: whether two readers agree on what a group of papers is about."""

from __future__ import annotations

import pytest

from magnetor.naming import (
    ACCEPTS,
    HUMAN,
    PARTIAL,
    REJECTS,
    agreement,
    freeze,
    list_sets,
    load_set,
    names,
    to_judge,
    to_name,
    write_judgement,
    write_name,
)

_CLUSTERS = [
    (("P1", "P2", "P3"), ("decoherence", "qubits")),
    (("P4", "P5"), ("homology", "manifolds")),
]


def _set(tmp_path, clusters=None):
    return freeze("qm", clusters or _CLUSTERS, naming_dir=tmp_path)


def test_freezing_the_same_grouping_twice_gives_the_same_set(tmp_path) -> None:
    """Otherwise two readers a week apart would be naming different things."""
    first = _set(tmp_path)
    second = _set(tmp_path)
    assert first.set_id == second.set_id
    assert len(list_sets(naming_dir=tmp_path)) == 1


def test_a_different_grouping_is_a_different_set(tmp_path) -> None:
    other = _set(tmp_path, [(("P9",), ("other",))])
    assert other.set_id != _set(tmp_path).set_id


def test_a_frozen_set_survives_a_reload(tmp_path) -> None:
    saved = _set(tmp_path)
    loaded = load_set(saved.set_id, naming_dir=tmp_path)
    assert loaded is not None
    assert loaded.clusters == saved.clusters
    assert loaded.domain == "qm"


def test_naming_never_shows_the_computed_words(tmp_path) -> None:
    """A reader shown the word label would paraphrase it, not read the papers.

    The snapshot keeps the label for later comparison; this asserts the naming
    pass hands back papers, and that the reader is asked about those.
    """
    naming_set = _set(tmp_path)
    pending = to_name(naming_set, coder="ada", naming_dir=tmp_path)
    assert [c.paper_ids for c in pending] == [("P1", "P2", "P3"), ("P4", "P5")]


def test_a_reader_is_only_offered_clusters_they_have_not_named(tmp_path) -> None:
    naming_set = _set(tmp_path)
    first = naming_set.clusters[0]
    write_name(naming_set, first.cluster_id, "Qubit decay", coder="ada",
               naming_dir=tmp_path)
    left = to_name(naming_set, coder="ada", naming_dir=tmp_path)
    assert [c.cluster_id for c in left] == [naming_set.clusters[1].cluster_id]
    # The other reader has done nothing, so they still see both.
    assert len(to_name(naming_set, coder="grace", naming_dir=tmp_path)) == 2


def test_one_reader_never_sees_another_reader_s_name(tmp_path) -> None:
    """The blindness is structural: the naming pass returns clusters, not names."""
    naming_set = _set(tmp_path)
    write_name(naming_set, naming_set.clusters[0].cluster_id, "Qubit decay",
               coder="ada", naming_dir=tmp_path)
    for cluster in to_name(naming_set, coder="grace", naming_dir=tmp_path):
        assert "Qubit decay" not in str(cluster)


def test_a_name_for_a_cluster_outside_the_set_is_refused(tmp_path) -> None:
    naming_set = _set(tmp_path)
    with pytest.raises(ValueError, match="not in set"):
        write_name(naming_set, "nonsense", "x", coder="ada", naming_dir=tmp_path)


def test_an_empty_name_is_refused(tmp_path) -> None:
    naming_set = _set(tmp_path)
    with pytest.raises(ValueError, match="cannot be empty"):
        write_name(naming_set, naming_set.clusters[0].cluster_id, "   ",
                   coder="ada", naming_dir=tmp_path)


@pytest.mark.parametrize("invisible", ["﻿", "​", "﻿  ‍"])
def test_an_invisible_name_is_refused(tmp_path, invisible: str) -> None:
    """A byte-order mark is not whitespace, so ``strip`` leaves it and it reads
    as a real answer. Piping into the CLI recorded one as a cluster's name."""
    naming_set = _set(tmp_path)
    with pytest.raises(ValueError, match="cannot be empty"):
        write_name(naming_set, naming_set.clusters[0].cluster_id, invisible,
                   coder="ada", naming_dir=tmp_path)


def test_an_invisible_character_is_stripped_from_a_real_name(tmp_path) -> None:
    """A pasted name keeps its words; only the invisible prefix goes."""
    naming_set = _set(tmp_path)
    written = write_name(naming_set, naming_set.clusters[0].cluster_id,
                         "﻿Qubit decoherence", coder="ada", naming_dir=tmp_path)
    assert written.text == "Qubit decoherence"


def test_names_are_appended_never_overwritten(tmp_path) -> None:
    """Disagreement is the data; an overwrite would destroy it."""
    naming_set = _set(tmp_path)
    cid = naming_set.clusters[0].cluster_id
    write_name(naming_set, cid, "First try", coder="ada", naming_dir=tmp_path)
    write_name(naming_set, cid, "Second try", coder="ada", naming_dir=tmp_path)
    stored = [n.text for n in names(naming_set.set_id, naming_dir=tmp_path)]
    assert stored == ["First try", "Second try"]


def test_a_candidate_carries_no_author(tmp_path) -> None:
    """Stage 2 depends on this: a machine name must be indistinguishable."""
    naming_set = _set(tmp_path)
    write_name(naming_set, naming_set.clusters[0].cluster_id, "Qubit decay",
               coder="ada", naming_dir=tmp_path)
    candidate = to_judge(naming_set, judge="grace", naming_dir=tmp_path)[0]
    assert not hasattr(candidate, "coder")
    assert not hasattr(candidate, "source")


def test_a_judge_rules_on_their_own_name_too(tmp_path) -> None:
    """Judging only other people's names identifies them by elimination."""
    naming_set = _set(tmp_path)
    cid = naming_set.clusters[0].cluster_id
    write_name(naming_set, cid, "Ada's name", coder="ada", naming_dir=tmp_path)
    write_name(naming_set, cid, "Grace's name", coder="grace", naming_dir=tmp_path)
    texts = {c.text for c in to_judge(naming_set, judge="ada", naming_dir=tmp_path)}
    assert texts == {"Ada's name", "Grace's name"}


def test_judges_are_given_different_orders(tmp_path) -> None:
    """Position must not leak authorship — everyone wrote their own first."""
    clusters = [((f"P{i}",), ("w",)) for i in range(12)]
    naming_set = _set(tmp_path, clusters)
    for cluster in naming_set.clusters:
        write_name(naming_set, cluster.cluster_id, f"n{cluster.cluster_id[:4]}",
                   coder="ada", naming_dir=tmp_path)
    ada = [c.name_id for c in to_judge(naming_set, judge="ada", naming_dir=tmp_path)]
    grace = [c.name_id for c in to_judge(naming_set, judge="grace", naming_dir=tmp_path)]
    assert sorted(ada) == sorted(grace)
    assert ada != grace


def test_the_shuffle_is_stable_so_a_pass_can_be_resumed(tmp_path) -> None:
    clusters = [((f"P{i}",), ("w",)) for i in range(12)]
    naming_set = _set(tmp_path, clusters)
    for cluster in naming_set.clusters:
        write_name(naming_set, cluster.cluster_id, f"n{cluster.cluster_id[:4]}",
                   coder="ada", naming_dir=tmp_path)
    first = [c.name_id for c in to_judge(naming_set, judge="ada", naming_dir=tmp_path)]
    second = [c.name_id for c in to_judge(naming_set, judge="ada", naming_dir=tmp_path)]
    assert first == second


def test_a_judge_is_not_asked_twice(tmp_path) -> None:
    naming_set = _set(tmp_path)
    write_name(naming_set, naming_set.clusters[0].cluster_id, "A name",
               coder="ada", naming_dir=tmp_path)
    candidate = to_judge(naming_set, judge="grace", naming_dir=tmp_path)[0]
    write_judgement(naming_set, candidate, ACCEPTS, judge="grace", naming_dir=tmp_path)
    assert to_judge(naming_set, judge="grace", naming_dir=tmp_path) == ()


def test_an_unknown_verdict_is_refused(tmp_path) -> None:
    naming_set = _set(tmp_path)
    write_name(naming_set, naming_set.clusters[0].cluster_id, "A name",
               coder="ada", naming_dir=tmp_path)
    candidate = to_judge(naming_set, judge="grace", naming_dir=tmp_path)[0]
    with pytest.raises(ValueError, match="verdict must be one of"):
        write_judgement(naming_set, candidate, "maybe", judge="grace",
                        naming_dir=tmp_path)


def test_one_judge_cannot_produce_an_agreement_figure(tmp_path) -> None:
    """A single reader's consistency is not agreement, and must not read as it."""
    naming_set = _set(tmp_path)
    for cluster in naming_set.clusters:
        write_name(naming_set, cluster.cluster_id, "A name", coder="ada",
                   naming_dir=tmp_path)
    for candidate in to_judge(naming_set, judge="ada", naming_dir=tmp_path):
        write_judgement(naming_set, candidate, ACCEPTS, judge="ada",
                        naming_dir=tmp_path)
    result = agreement(naming_set.set_id, naming_dir=tmp_path)
    assert not result.measurable
    assert result.alpha is None
    assert "two people" in result.summary()


def _two_judges(tmp_path, verdicts_a, verdicts_b):
    """Rule on the same names with both judges.

    Verdicts are assigned by name identity, not by position: each judge sees a
    different order, so zipping to the shuffled list would pair a verdict with
    whichever name happened to land there.
    """
    clusters = [((f"P{i}",), ("w",)) for i in range(len(verdicts_a))]
    naming_set = freeze("qm", clusters, naming_dir=tmp_path)
    for cluster in naming_set.clusters:
        write_name(naming_set, cluster.cluster_id, f"name {cluster.cluster_id[:4]}",
                   coder="ada", naming_dir=tmp_path)
    order = [n.name_id for n in names(naming_set.set_id, naming_dir=tmp_path)]
    for who, verdicts in (("ada", verdicts_a), ("grace", verdicts_b)):
        plan = dict(zip(order, verdicts, strict=True))
        for candidate in to_judge(naming_set, judge=who, naming_dir=tmp_path):
            write_judgement(naming_set, candidate, plan[candidate.name_id],
                            judge=who, naming_dir=tmp_path)
    return naming_set


def test_two_judges_who_never_agree_score_below_zero(tmp_path) -> None:
    naming_set = _two_judges(
        tmp_path,
        [ACCEPTS, REJECTS, ACCEPTS, REJECTS],
        [REJECTS, ACCEPTS, REJECTS, ACCEPTS],
    )
    result = agreement(naming_set.set_id, naming_dir=tmp_path)
    assert result.measurable
    # Systematic opposition scores worse than chance, not merely zero.
    assert result.alpha is not None and result.alpha < 0.0
    assert result.judges == 2
    assert result.units == 4


def test_two_judges_who_always_agree_score_one(tmp_path) -> None:
    """The positive control: without it, a broken alpha could pass every test."""
    naming_set = _two_judges(
        tmp_path,
        [ACCEPTS, REJECTS, PARTIAL, ACCEPTS],
        [ACCEPTS, REJECTS, PARTIAL, ACCEPTS],
    )
    result = agreement(naming_set.set_id, naming_dir=tmp_path)
    assert result.alpha == 1.0


def test_partial_is_recorded_as_its_own_answer(tmp_path) -> None:
    """Collapsing partial into accept or reject would hide the informative case."""
    naming_set = _two_judges(
        tmp_path, [ACCEPTS, PARTIAL, REJECTS], [ACCEPTS, PARTIAL, REJECTS]
    )
    result = agreement(naming_set.set_id, naming_dir=tmp_path)
    assert result.verdict_spread[PARTIAL] == 2


def test_a_judge_who_changes_their_mind_is_counted_once(tmp_path) -> None:
    """Counting both rulings would inflate a disagreement they just resolved."""
    naming_set = _set(tmp_path, [(("P1",), ("w",))])
    cid = naming_set.clusters[0].cluster_id
    write_name(naming_set, cid, "A name", coder="ada", naming_dir=tmp_path)
    candidate = to_judge(naming_set, judge="ada", naming_dir=tmp_path)[0]
    write_judgement(naming_set, candidate, REJECTS, judge="ada", naming_dir=tmp_path)
    write_judgement(naming_set, candidate, ACCEPTS, judge="ada", naming_dir=tmp_path)
    write_judgement(naming_set, candidate, ACCEPTS, judge="grace", naming_dir=tmp_path)
    result = agreement(naming_set.set_id, naming_dir=tmp_path)
    assert result.verdict_spread == {ACCEPTS: 2}


def test_names_default_to_being_human_written(tmp_path) -> None:
    """Stage 2 adds machine names to the same pool; the default must not be it."""
    naming_set = _set(tmp_path)
    written = write_name(naming_set, naming_set.clusters[0].cluster_id, "A name",
                         coder="ada", naming_dir=tmp_path)
    assert written.source == HUMAN


def test_a_corrupt_line_does_not_lose_the_rest_of_the_log(tmp_path) -> None:
    naming_set = _set(tmp_path)
    write_name(naming_set, naming_set.clusters[0].cluster_id, "Kept",
               coder="ada", naming_dir=tmp_path)
    log = tmp_path / f"{naming_set.set_id}.jsonl"
    log.write_text("{ not json\n" + log.read_text(encoding="utf-8"), encoding="utf-8")
    assert [n.text for n in names(naming_set.set_id, naming_dir=tmp_path)] == ["Kept"]
