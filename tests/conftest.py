"""Shared fixtures and fakes for the test suite."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

import pytest

from magnetor.types import Domain, Paper


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    """Point every test's storage at a temp dir, never the real /data.

    Also unset the Voyage key so no test can accidentally hit the live API.
    """
    monkeypatch.setenv("MAGNETOR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("MAGNETOR_VOYAGE_API_KEY", raising=False)
    # config.DATA_ROOT is read at import time, so patch the bound value too.
    import magnetor.config as config

    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    return tmp_path


def make_paper(
    *,
    domain: Domain = Domain.QUANTUM_MECHANICS,
    external_id: str = "2601.00001",
    doi: str | None = None,
    published: dt.datetime | None = None,
    abstract: str = "An abstract.",
) -> Paper:
    """Build a Paper with sensible defaults for tests."""
    return Paper(
        domain=domain,
        source="test",
        external_id=external_id,
        title="A Test Paper",
        abstract=abstract,
        authors=("Ada Lovelace",),
        published=published or dt.datetime(2026, 6, 20, tzinfo=dt.UTC),
        doi=doi,
    )


class FakeSource:
    """In-memory DomainSource for pipeline tests."""

    def __init__(self, domain: Domain, papers: list[Paper]) -> None:
        self._domain = domain
        self._papers = papers
        self.calls: list[tuple[dt.datetime | None, int]] = []

    @property
    def domain(self) -> Domain:
        return self._domain

    def fetch(self, *, since: dt.datetime | None, limit: int) -> Iterable[Paper]:
        self.calls.append((since, limit))
        return list(self._papers[:limit])


class FakeEmbedder:
    """Deterministic Embedder for tests — no network.

    Maps each text to a fixed-length vector by hashing its words into buckets,
    so semantically-similar strings (shared words) land near each other.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for word in text.lower().split():
            vec[hash(word) % self._dimension] += 1.0
        # Guarantee a non-zero vector so cosine is well-defined.
        if not any(vec):
            vec[0] = 1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
