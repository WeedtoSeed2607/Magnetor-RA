"""Per-domain vector index (Spec Section 10).

A brute-force cosine index backed by NumPy. The spec calls for a dedicated
vector store "once corpus size warrants it"; at current scale an exact scan is
instant and dependency-light, so this stays behind a small surface that a
sqlite-vec / FAISS backend can later implement without touching callers.

Vectors are L2-normalised on the way in, so cosine similarity is a plain dot
product. One index lives inside one domain's directory — it never mixes
vectors across domains, upholding the isolation invariant.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from magnetor.errors import MagnetorError
from magnetor.types import Domain

#: Conventional filename for a domain's on-disk index.
VECTOR_FILENAME = "vectors.npz"

Vector = Sequence[float]


class VectorIndex:
    """In-memory brute-force cosine index for one domain, persisted as ``.npz``."""

    def __init__(self, domain: Domain, path: Path, dimension: int) -> None:
        self._domain = domain
        self._path = Path(path)
        self._dim = dimension
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._matrix: NDArray[np.float32] = np.empty((0, dimension), dtype=np.float32)
        if self._path.exists():
            self._load()

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def dimension(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> set[str]:
        """External ids currently indexed (for incremental embedding)."""
        return set(self._id_set)

    def upsert(self, external_id: str, vector: Vector) -> bool:
        """Add one vector; return ``False`` if ``external_id`` is already present."""
        return self.upsert_many([(external_id, vector)]) == 1

    def upsert_many(self, items: Iterable[tuple[str, Vector]]) -> int:
        """Add many vectors, skipping ids already present. Returns count added."""
        new_rows: list[NDArray[np.float32]] = []
        added = 0
        for external_id, vector in items:
            if external_id in self._id_set:
                continue
            new_rows.append(self._prepare(vector))
            self._ids.append(external_id)
            self._id_set.add(external_id)
            added += 1
        if new_rows:
            self._matrix = np.vstack([self._matrix, np.vstack(new_rows)])
        return added

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(external_id, cosine)`` pairs, most similar first."""
        if not self._ids or k <= 0:
            return []
        q = self._prepare(query)
        scores = self._matrix @ q
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in top]

    def save(self) -> None:
        """Persist the index to its ``.npz`` file (creates parent dirs)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._path,
            matrix=self._matrix,
            ids=np.asarray(self._ids, dtype=np.str_),
            dim=np.asarray(self._dim),
        )

    def _load(self) -> None:
        with np.load(self._path) as data:
            dim = int(data["dim"])
            if dim != self._dim:
                raise MagnetorError(
                    f"Vector index {self._path} has dimension {dim}, expected {self._dim}"
                )
            self._matrix = data["matrix"].astype(np.float32)
            self._ids = [str(x) for x in data["ids"].tolist()]
            self._id_set = set(self._ids)

    def _prepare(self, vector: Vector) -> NDArray[np.float32]:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self._dim,):
            raise MagnetorError(
                f"Expected a length-{self._dim} vector, got shape {arr.shape}"
            )
        return _normalise(arr)


def stored_count(path: Path) -> int:
    """Number of vectors persisted at ``path``, or 0 if it does not exist.

    Reads only the id array, so status checks stay cheap and need no embedder.
    """
    if not path.exists():
        return 0
    with np.load(path) as data:
        return len(data["ids"])


def _normalise(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the unit-length version of ``vector`` (zero vector left as-is)."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    result: NDArray[np.float32] = (vector / norm).astype(np.float32)
    return result
