"""Sharing the experiment: what the public bundle carries, and the round trip back."""

from __future__ import annotations

import json

from magnetor.naming import (
    ACCEPTS,
    agreement,
    bundle,
    export_lines,
    freeze,
    import_lines,
    judgements,
    names,
    read_bundle,
    to_judge,
    write_judgement,
    write_name,
)

_CLUSTERS = [
    (("P1", "P2"), ("decoherence", "qubits")),
    (("P3",), ("homology",)),
]
_TITLES = {"P1": "Qubit decay", "P2": "Cavity coupling", "P3": "Manifold homology"}


def _set(tmp_path):
    return freeze("qm", _CLUSTERS, naming_dir=tmp_path)


def test_the_bundle_carries_titles_and_membership(tmp_path) -> None:
    payload = bundle(_set(tmp_path), _TITLES)
    assert len(payload["clusters"]) == 2
    assert payload["titles"]["P1"] == "Qubit decay"


def test_the_bundle_never_carries_the_computed_words(tmp_path) -> None:
    """Shipping them would put the words one click from the reader who must not see them."""
    payload = bundle(_set(tmp_path), _TITLES)
    assert "decoherence" not in json.dumps(payload)
    assert all("computed_label" not in c for c in payload["clusters"])


def test_a_bundle_survives_the_round_trip(tmp_path) -> None:
    original = _set(tmp_path)
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle(original, _TITLES)), encoding="utf-8")
    restored, titles = read_bundle(path)
    assert restored.set_id == original.set_id
    assert [c.paper_ids for c in restored.clusters] == [
        c.paper_ids for c in original.clusters
    ]
    assert titles == _TITLES


def test_a_restored_cluster_shows_no_label_rather_than_an_empty_one(tmp_path) -> None:
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle(_set(tmp_path), _TITLES)), encoding="utf-8")
    restored, _ = read_bundle(path)
    assert all(c.computed_label == () for c in restored.clusters)


def test_a_reader_exports_only_their_own_work(tmp_path) -> None:
    """Otherwise a remote reader carries home the answers they must not see."""
    naming_set = _set(tmp_path)
    for cluster in naming_set.clusters:
        write_name(naming_set, cluster.cluster_id, f"ada {cluster.cluster_id[:4]}",
                   coder="ada", naming_dir=tmp_path)
        write_name(naming_set, cluster.cluster_id, f"grace {cluster.cluster_id[:4]}",
                   coder="grace", naming_dir=tmp_path)
    exported = export_lines(naming_set.set_id, "grace", naming_dir=tmp_path)
    assert "grace" in exported
    assert "ada" not in exported


def test_exported_judgements_travel_too(tmp_path) -> None:
    naming_set = _set(tmp_path)
    write_name(naming_set, naming_set.clusters[0].cluster_id, "A name",
               coder="ada", naming_dir=tmp_path)
    candidate = to_judge(naming_set, judge="grace", naming_dir=tmp_path)[0]
    write_judgement(naming_set, candidate, ACCEPTS, judge="grace", naming_dir=tmp_path)
    exported = export_lines(naming_set.set_id, "grace", naming_dir=tmp_path)
    assert "judgement" in exported


def test_importing_merges_a_remote_reader(tmp_path, tmp_path_factory) -> None:
    """The whole point: the hosted app's answers become part of the local figure."""
    remote = tmp_path_factory.mktemp("remote")
    away = freeze("qm", _CLUSTERS, naming_dir=remote)
    for cluster in away.clusters:
        write_name(away, cluster.cluster_id, f"grace {cluster.cluster_id[:4]}",
                   coder="grace", naming_dir=remote)
    carried = export_lines(away.set_id, "grace", naming_dir=remote)

    home = _set(tmp_path)
    assert home.set_id == away.set_id  # same membership, same identity
    added, skipped = import_lines(home.set_id, carried, naming_dir=tmp_path)
    assert (added, skipped) == (2, 0)
    assert {n.coder for n in names(home.set_id, naming_dir=tmp_path)} == {"grace"}


def test_importing_the_same_file_twice_changes_nothing(tmp_path, tmp_path_factory) -> None:
    """A reader who sends their answers twice must not count twice."""
    remote = tmp_path_factory.mktemp("remote")
    away = freeze("qm", _CLUSTERS, naming_dir=remote)
    write_name(away, away.clusters[0].cluster_id, "grace one", coder="grace",
               naming_dir=remote)
    carried = export_lines(away.set_id, "grace", naming_dir=remote)

    home = _set(tmp_path)
    import_lines(home.set_id, carried, naming_dir=tmp_path)
    added, skipped = import_lines(home.set_id, carried, naming_dir=tmp_path)
    assert (added, skipped) == (0, 1)
    assert len(names(home.set_id, naming_dir=tmp_path)) == 1


def test_unreadable_or_foreign_lines_are_skipped_not_stored(tmp_path) -> None:
    home = _set(tmp_path)
    added, skipped = import_lines(
        home.set_id,
        '{ not json\n{"kind": "something-else", "x": "1"}\n\n',
        naming_dir=tmp_path,
    )
    assert (added, skipped) == (0, 2)
    assert names(home.set_id, naming_dir=tmp_path) == ()
    assert judgements(home.set_id, naming_dir=tmp_path) == ()


def test_a_merged_remote_reader_reaches_the_figure(tmp_path, tmp_path_factory) -> None:
    remote = tmp_path_factory.mktemp("remote")
    away = freeze("qm", _CLUSTERS, naming_dir=remote)
    home = _set(tmp_path)
    for cluster in home.clusters:
        write_name(home, cluster.cluster_id, f"ada {cluster.cluster_id[:4]}",
                   coder="ada", naming_dir=tmp_path)
    carried_names = export_lines(home.set_id, "ada", naming_dir=tmp_path)
    import_lines(away.set_id, carried_names, naming_dir=remote)

    for who, directory in (("ada", tmp_path), ("grace", remote)):
        for candidate in to_judge(home, judge=who, naming_dir=directory):
            write_judgement(home, candidate, ACCEPTS, judge=who, naming_dir=directory)
    import_lines(
        home.set_id, export_lines(away.set_id, "grace", naming_dir=remote),
        naming_dir=tmp_path,
    )
    result = agreement(home.set_id, naming_dir=tmp_path)
    assert result.judges == 2
    assert result.units == 2
