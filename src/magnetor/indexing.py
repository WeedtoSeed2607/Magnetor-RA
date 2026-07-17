"""Embedding orchestration (Phase 2, Step 3).

Ties a domain's stored records, its vector index, and an embedder together:

    read records -> skip already-indexed / empty -> embed passages -> upsert -> save

Embedder, store, and index are injected, so this module never touches the
network directly and is fully unit-testable with a fake embedder. It stays
within one domain: records, embeddings, and the index all belong to that domain,
upholding the isolation invariant.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from magnetor.config import DomainConfig
from magnetor.embeddings.base import Embedder
from magnetor.resources import DomainStore
from magnetor.types import Domain, Paper
from magnetor.vectors import VECTOR_FILENAME, VectorIndex

DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Outcome of one embedding run for one domain."""

    domain: Domain
    records: int
    embedded: int
    skipped_existing: int
    skipped_empty: int
    index_size: int


def open_index(config: DomainConfig, embedder: Embedder) -> VectorIndex:
    """Open the domain's vector index, sized to the embedder's dimension.

    The index file lives inside the domain directory, so it never mixes vectors
    across domains.
    """
    path = Path(config.storage_dir) / VECTOR_FILENAME
    return VectorIndex(config.domain, path, embedder.dimension)


def run_embedding(
    store: DomainStore,
    index: VectorIndex,
    embedder: Embedder,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingResult:
    """Embed stored records not yet in the index and persist the result.

    Idempotent: records whose ``external_id`` is already indexed are skipped, so
    re-running only embeds new material. Records with no embeddable text (no
    title and no abstract) are counted and skipped rather than embedded.
    """
    records = store.read_records()
    existing = index.ids()

    pending: list[tuple[str, str]] = []
    skipped_existing = 0
    skipped_empty = 0
    for paper in records:
        if paper.external_id in existing:
            skipped_existing += 1
            continue
        text = _embed_text(paper)
        if not text:
            skipped_empty += 1
            continue
        pending.append((paper.external_id, text))

    embedded = 0
    for batch in _batched(pending, batch_size):
        ids = [external_id for external_id, _ in batch]
        vectors = embedder.embed_documents([text for _, text in batch])
        index.upsert_many(zip(ids, vectors, strict=True))
        embedded += len(batch)

    if embedded:
        index.save()

    return EmbeddingResult(
        domain=store.domain,
        records=len(records),
        embedded=embedded,
        skipped_existing=skipped_existing,
        skipped_empty=skipped_empty,
        index_size=len(index),
    )


def _embed_text(paper: Paper) -> str:
    """Text used to represent a paper: title and abstract, whichever exist."""
    parts = [paper.title.strip(), paper.abstract.strip()]
    return "\n\n".join(part for part in parts if part)


def _batched(items: list[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]
