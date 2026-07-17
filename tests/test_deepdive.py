"""Branch B path selection (Spec 7.2): Anchor-Lock vs Field-Map + grounded render."""

from __future__ import annotations

from collections.abc import Sequence

from magnetor.citations import Citation
from magnetor.config import get_domain_config
from magnetor.deepdive import Path, build_deep_dive, render_grounded_context
from magnetor.resources import DomainStore
from magnetor.router import CrossDomainRouter
from magnetor.types import Domain, Paper
from magnetor.vectors import VectorIndex
from tests.conftest import make_paper

_QM = Domain.QUANTUM_MECHANICS


class _FixedEmbedder:
    """Embedder whose query vector is fixed, so cosine scores are controllable."""

    dimension = 3

    def __init__(self, query_vector: Sequence[float]) -> None:
        self._q = list(query_vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._q)


class _FakeExpander:
    def __init__(self, forward: Sequence[Citation] = (), backward: Sequence[Citation] = ()) -> None:
        self._forward = list(forward)
        self._backward = list(backward)
        self.calls: list[Paper] = []

    def expand(self, paper: Paper) -> tuple[list[Citation], list[Citation]]:
        self.calls.append(paper)
        return list(self._forward), list(self._backward)


def _store(tmp_path, papers: list[Paper]) -> None:
    store = DomainStore(_QM, get_domain_config(_QM).storage_dir)
    for paper in papers:
        store.store(paper)


def _router(tmp_path, vectors: list[tuple[str, list[float]]]) -> CrossDomainRouter:
    index = VectorIndex(_QM, tmp_path / "qm.npz", dimension=3)
    for external_id, vec in vectors:
        index.upsert(external_id, vec)
    return CrossDomainRouter({_QM: index})


def test_anchor_lock_above_threshold(tmp_path) -> None:
    _store(tmp_path, [make_paper(external_id="a", abstract="quantum spin")])
    router = _router(tmp_path, [("a", [1.0, 0.0, 0.0])])  # cosine 1.0 >= 0.55
    expander = _FakeExpander(forward=[Citation("Cites A", 2026)])

    result = build_deep_dive(
        _FixedEmbedder([1.0, 0.0, 0.0]), router, "q", expander, domain=_QM, threshold=0.5
    )

    assert result.path is Path.ANCHOR_LOCK
    assert result.anchor is not None
    assert result.anchor.paper.external_id == "a"
    assert [c.title for c in result.anchor.forward] == ["Cites A"]
    assert expander.calls  # citation expansion happened


def test_field_map_below_threshold(tmp_path) -> None:
    papers = [make_paper(external_id=f"p{i}", abstract=f"pos {i}") for i in range(3)]
    _store(tmp_path, papers)
    router = _router(
        tmp_path,
        [("p0", [0.45, 0.893, 0.0]), ("p1", [0.42, 0.9075, 0.0]), ("p2", [0.40, 0.9165, 0.0])],
    )  # all cosine < 0.55 vs [1,0,0]
    expander = _FakeExpander()

    result = build_deep_dive(
        _FixedEmbedder([1.0, 0.0, 0.0]), router, "q", expander, domain=_QM, threshold=0.5
    )

    assert result.path is Path.FIELD_MAP
    assert result.field_map is not None
    assert [pos.rank for pos in result.field_map.positions] == [1, 2, 3]
    assert result.field_map.positions[0].paper.external_id == "p0"  # highest score first
    assert not expander.calls  # no citation expansion on the fallback path


def test_field_map_caps_at_five(tmp_path) -> None:
    papers = [make_paper(external_id=f"p{i}", abstract=f"pos {i}") for i in range(7)]
    _store(tmp_path, papers)
    vectors = [(f"p{i}", [0.4, 0.9165, 0.0]) for i in range(7)]
    router = _router(tmp_path, vectors)

    result = build_deep_dive(
        _FixedEmbedder([1.0, 0.0, 0.0]), router, "q", _FakeExpander(),
        domain=_QM, k=7, threshold=0.5,
    )
    assert result.path is Path.FIELD_MAP
    assert result.field_map is not None
    assert len(result.field_map.positions) == 5


def test_no_results(tmp_path) -> None:
    router = CrossDomainRouter({_QM: VectorIndex(_QM, tmp_path / "qm.npz", 3)})
    result = build_deep_dive(
        _FixedEmbedder([1.0, 0.0, 0.0]), router, "q", _FakeExpander(), domain=_QM
    )
    assert result.path is None


def test_grounded_context_marks_immutable_sections(tmp_path) -> None:
    _store(tmp_path, [make_paper(external_id="a", abstract="quantum spin entanglement")])
    router = _router(tmp_path, [("a", [1.0, 0.0, 0.0])])
    result = build_deep_dive(
        _FixedEmbedder([1.0, 0.0, 0.0]), router, "what is spin?",
        _FakeExpander(forward=[Citation("Cites A", 2026, doi="10.1/x")]),
        domain=_QM, threshold=0.5,
    )

    context = render_grounded_context(result)
    assert "[GROUNDED" in context
    assert "MUST NOT" in context
    assert "SYNTHESIS SLOT" in context
    assert "quantum spin entanglement" in context  # abstract present verbatim
    assert "Cites A" in context
    # The concrete decision is stated for the model.
    assert "ANCHOR PAPER [GROUNDED" in context
