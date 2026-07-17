"""CrossDomainRouter: centroid scoring, margin selection, retrieve, logging."""

from __future__ import annotations

import json

from magnetor.router import CrossDomainRouter, retrieve
from magnetor.types import Domain
from magnetor.vectors import VectorIndex
from tests.conftest import FakeEmbedder


def _index(tmp_path, domain, vectors) -> VectorIndex:
    index = VectorIndex(domain, tmp_path / f"{domain.value}.npz", dimension=3)
    for i, vec in enumerate(vectors):
        index.upsert(f"{domain.value}-{i}", vec)
    return index


def test_routes_to_clear_winner(tmp_path) -> None:
    indices = {
        Domain.QUANTUM_MECHANICS: _index(tmp_path, Domain.QUANTUM_MECHANICS, [[1.0, 0.0, 0.0]]),
        Domain.MATHEMATICS: _index(tmp_path, Domain.MATHEMATICS, [[0.0, 1.0, 0.0]]),
    }
    router = CrossDomainRouter(indices, margin=0.05)
    routing = router.route([1.0, 0.0, 0.0])
    assert routing.selected == (Domain.QUANTUM_MECHANICS,)
    assert routing.scored[0].domain is Domain.QUANTUM_MECHANICS


def test_routes_to_top_two_within_margin(tmp_path) -> None:
    # Two domains whose centroids are both close to the query -> interdisciplinary.
    indices = {
        Domain.QUANTUM_MECHANICS: _index(tmp_path, Domain.QUANTUM_MECHANICS, [[1.0, 0.1, 0.0]]),
        Domain.MATHEMATICS: _index(tmp_path, Domain.MATHEMATICS, [[1.0, 0.0, 0.1]]),
        Domain.HISTORY: _index(tmp_path, Domain.HISTORY, [[0.0, 0.0, 1.0]]),
    }
    router = CrossDomainRouter(indices, margin=0.1)
    routing = router.route([1.0, 0.05, 0.05])
    assert len(routing.selected) == 2
    assert set(routing.selected) == {Domain.QUANTUM_MECHANICS, Domain.MATHEMATICS}


def test_representative_scoring_beats_centroid_dilution(tmp_path) -> None:
    # Target domain: a strong 3-paper cluster on the query plus 5 unrelated
    # papers. A single centroid would be dragged toward the bulk and lose to the
    # distractor; top-K scoring captures the relevant cluster and wins.
    # (Vectors are normalised on store, so the distractor must be directionally
    # off-axis — a smaller magnitude alone would normalise to a perfect match.)
    target = [[1.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 1.0]] * 5
    distractor = [[0.75, 0.6614, 0.0]]  # cosine ~0.75 with the query
    indices = {
        Domain.QUANTUM_MECHANICS: _index(tmp_path, Domain.QUANTUM_MECHANICS, target),
        Domain.MATHEMATICS: _index(tmp_path, Domain.MATHEMATICS, distractor),
    }
    routing = CrossDomainRouter(indices, sample_size=3).route([1.0, 0.0, 0.0])
    assert routing.selected[0] is Domain.QUANTUM_MECHANICS
    assert routing.scored[0].score > routing.scored[1].score


def test_skips_domains_without_vectors(tmp_path) -> None:
    indices = {
        Domain.QUANTUM_MECHANICS: _index(tmp_path, Domain.QUANTUM_MECHANICS, [[1.0, 0.0, 0.0]]),
        # empty index -> centroid None -> not scored
        Domain.MATHEMATICS: VectorIndex(Domain.MATHEMATICS, tmp_path / "m.npz", 3),
    }
    router = CrossDomainRouter(indices, margin=0.05)
    routing = router.route([1.0, 0.0, 0.0])
    assert [s.domain for s in routing.scored] == [Domain.QUANTUM_MECHANICS]
    assert routing.selected == (Domain.QUANTUM_MECHANICS,)


def test_empty_everything_selects_nothing(tmp_path) -> None:
    empty = VectorIndex(Domain.QUANTUM_MECHANICS, tmp_path / "q.npz", 3)
    indices = {Domain.QUANTUM_MECHANICS: empty}
    routing = CrossDomainRouter(indices).route([1.0, 0.0, 0.0])
    assert routing.selected == ()
    assert routing.scored == ()


def test_retrieve_returns_hits_from_selected_domain(tmp_path) -> None:
    store_index = _index(tmp_path, Domain.QUANTUM_MECHANICS, [[1.0, 0.0, 0.0], [0.2, 1.0, 0.0]])
    indices = {Domain.QUANTUM_MECHANICS: store_index}
    router = CrossDomainRouter(indices)

    class _Fixed:
        dimension = 3

        def embed_documents(self, texts):  # pragma: no cover - unused here
            return [[0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    routing, hits = retrieve(_Fixed(), router, "q", k=2)
    assert routing is not None
    assert hits[0].domain is Domain.QUANTUM_MECHANICS
    assert hits[0].external_id == "qm-0"  # closest to [1,0,0]
    assert hits[0].score >= hits[1].score


def test_retrieve_honors_forced_domain(tmp_path) -> None:
    indices = {
        Domain.QUANTUM_MECHANICS: _index(tmp_path, Domain.QUANTUM_MECHANICS, [[1.0, 0.0, 0.0]]),
        Domain.MATHEMATICS: _index(tmp_path, Domain.MATHEMATICS, [[0.0, 1.0, 0.0]]),
    }
    router = CrossDomainRouter(indices)
    routing, hits = retrieve(
        FakeEmbedder(dimension=3), router, "q", k=5, domain=Domain.MATHEMATICS
    )
    assert routing is None  # routing skipped when domain forced
    assert {hit.domain for hit in hits} == {Domain.MATHEMATICS}


def test_routing_decisions_are_logged(tmp_path) -> None:
    log_path = tmp_path / "routing_log.jsonl"
    qm_index = _index(tmp_path, Domain.QUANTUM_MECHANICS, [[1.0, 0.0, 0.0]])
    indices = {Domain.QUANTUM_MECHANICS: qm_index}
    router = CrossDomainRouter(indices, log_path=log_path)
    router.route([1.0, 0.0, 0.0], query="what is spin?")

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["query"] == "what is spin?"
    assert entry["selected"] == ["qm"]
    assert entry["scored"][0]["domain"] == "qm"
