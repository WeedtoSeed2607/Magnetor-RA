"""VectorIndex: cosine search, dedup, centroid, persistence, dim guard."""

from __future__ import annotations

import math

import pytest

from magnetor.errors import MagnetorError
from magnetor.types import Domain
from magnetor.vectors import VectorIndex, stored_count


def _index(tmp_path, dimension: int = 3) -> VectorIndex:
    return VectorIndex(Domain.QUANTUM_MECHANICS, tmp_path / "vectors.npz", dimension)


def test_search_returns_nearest_by_cosine(tmp_path) -> None:
    index = _index(tmp_path)
    index.upsert("x", [1.0, 0.0, 0.0])
    index.upsert("y", [0.0, 1.0, 0.0])
    index.upsert("z", [0.9, 0.1, 0.0])

    results = index.search([1.0, 0.0, 0.0], k=2)
    assert [external_id for external_id, _ in results] == ["x", "z"]
    # Cosine of the exact match is 1.0.
    assert math.isclose(results[0][1], 1.0, abs_tol=1e-6)


def test_upsert_is_idempotent_by_id(tmp_path) -> None:
    index = _index(tmp_path)
    assert index.upsert("x", [1.0, 0.0, 0.0]) is True
    assert index.upsert("x", [0.0, 1.0, 0.0]) is False  # same id ignored
    assert len(index) == 1
    assert index.ids() == {"x"}


def test_upsert_many_counts_new_only(tmp_path) -> None:
    index = _index(tmp_path)
    index.upsert("x", [1.0, 0.0, 0.0])
    added = index.upsert_many([("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0])])
    assert added == 1
    assert len(index) == 2


def test_stored_count_reads_without_embedder(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    assert stored_count(path) == 0  # absent file
    index = VectorIndex(Domain.QUANTUM_MECHANICS, path, dimension=3)
    index.upsert("x", [1.0, 0.0, 0.0])
    index.upsert("y", [0.0, 1.0, 0.0])
    index.save()
    assert stored_count(path) == 2


def test_persistence_roundtrip(tmp_path) -> None:
    index = _index(tmp_path)
    index.upsert("x", [1.0, 0.0, 0.0])
    index.upsert("y", [0.0, 2.0, 0.0])  # non-unit input, normalised on store
    index.save()

    reopened = _index(tmp_path)
    assert reopened.ids() == {"x", "y"}
    assert len(reopened) == 2
    # Search still finds the right neighbour after reload.
    assert reopened.search([0.0, 1.0, 0.0], k=1)[0][0] == "y"


def test_empty_search_returns_empty(tmp_path) -> None:
    assert _index(tmp_path).search([1.0, 0.0, 0.0], k=5) == []


def test_wrong_dimension_rejected(tmp_path) -> None:
    index = _index(tmp_path, dimension=3)
    with pytest.raises(MagnetorError):
        index.upsert("bad", [1.0, 0.0])  # length 2, expected 3


def test_dimension_mismatch_on_load_rejected(tmp_path) -> None:
    index = _index(tmp_path, dimension=3)
    index.upsert("x", [1.0, 0.0, 0.0])
    index.save()
    with pytest.raises(MagnetorError):
        VectorIndex(Domain.QUANTUM_MECHANICS, tmp_path / "vectors.npz", dimension=4)
