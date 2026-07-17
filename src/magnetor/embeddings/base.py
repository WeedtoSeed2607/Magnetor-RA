"""The embedding boundary.

External embedding providers are reached only through :class:`Embedder`, so the
router, vector index, and pipeline never import a concrete client and tests can
substitute a deterministic fake without any network (boundary-control rule).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Provider-agnostic text encoder with an asymmetric query/passage split."""

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages (e.g. abstracts) for indexing.

        Returns one vector per input, in the same order. An empty input yields
        an empty list.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...
