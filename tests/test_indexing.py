"""Embedding pipeline: idempotency, empty-text skipping, index population."""

from __future__ import annotations

from magnetor.indexing import run_embedding
from magnetor.resources import DomainStore
from magnetor.types import Domain
from magnetor.vectors import VectorIndex
from tests.conftest import FakeEmbedder, make_paper


def _store(tmp_path, domain=Domain.QUANTUM_MECHANICS) -> DomainStore:
    return DomainStore(domain, tmp_path / "data" / domain.value)


def _index(tmp_path, embedder, domain=Domain.QUANTUM_MECHANICS) -> VectorIndex:
    return VectorIndex(domain, tmp_path / "data" / domain.value / "vectors.npz", embedder.dimension)


def test_embeds_all_records_first_run(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="a", abstract="quantum spin entanglement"))
    store.store(make_paper(external_id="b", abstract="topology of manifolds"))
    embedder = FakeEmbedder()

    result = run_embedding(store, _index(tmp_path, embedder), embedder)

    assert result.records == 2
    assert result.embedded == 2
    assert result.skipped_existing == 0
    assert result.index_size == 2


def test_is_idempotent_across_runs(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="a", abstract="quantum spin"))
    embedder = FakeEmbedder()

    first = run_embedding(store, _index(tmp_path, embedder), embedder)
    assert first.embedded == 1

    # New instance reloads the persisted index; nothing new to embed.
    second = run_embedding(store, _index(tmp_path, embedder), embedder)
    assert second.embedded == 0
    assert second.skipped_existing == 1
    assert second.index_size == 1


def test_only_new_records_embedded_on_second_run(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="a", abstract="quantum spin"))
    embedder = FakeEmbedder()
    run_embedding(store, _index(tmp_path, embedder), embedder)

    store.store(make_paper(external_id="b", abstract="gauge theory"))
    result = run_embedding(store, _index(tmp_path, embedder), embedder)
    assert result.embedded == 1
    assert result.skipped_existing == 1
    assert result.index_size == 2


def test_records_without_text_are_skipped(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="a", abstract="has an abstract"))
    # Empty title AND abstract -> nothing to embed. Title default is non-empty,
    # so force both empty via a blank-title paper.
    blank = make_paper(external_id="b", abstract="")
    object.__setattr__(blank, "title", "")
    store.store(blank)
    embedder = FakeEmbedder()

    result = run_embedding(store, _index(tmp_path, embedder), embedder)
    assert result.embedded == 1
    assert result.skipped_empty == 1


def test_search_finds_semantically_closer_record(tmp_path) -> None:
    store = _store(tmp_path)
    store.store(make_paper(external_id="spin", abstract="quantum spin entanglement measurement"))
    store.store(make_paper(external_id="topo", abstract="algebraic topology homology groups"))
    embedder = FakeEmbedder()
    index = _index(tmp_path, embedder)
    run_embedding(store, index, embedder)

    query_vec = embedder.embed_query("quantum spin entanglement")
    top_id, _ = index.search(query_vec, k=1)[0]
    assert top_id == "spin"
