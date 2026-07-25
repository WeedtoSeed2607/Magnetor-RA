"""Branch C · L1 harvest: OpenAlex parsing, in-subgraph edges, caching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from magnetor.harvest import HarvestedPaper, run_harvest


class _FakeSource:
    """In-memory WorksSource. ``works`` is the fetchable universe; ``seed`` (if
    given) is what search() returns, so expansion can pull in works beyond it."""

    def __init__(
        self, works: list[dict[str, Any]], *, seed: list[dict[str, Any]] | None = None
    ) -> None:
        self._universe = {w["id"].rsplit("/", 1)[-1]: w for w in works}
        self._seed = seed if seed is not None else works
        self.calls = 0
        self.fetch_calls = 0

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        self.calls += 1
        return list(self._seed[:limit])

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        self.fetch_calls += 1
        return [self._universe[i] for i in ids if i in self._universe]


def _work(wid: str, refs: tuple[str, ...] = (), **kw: Any) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{wid}",
        "title": kw.get("title", f"Paper {wid}"),
        "display_name": f"Paper {wid}",
        "publication_year": kw.get("year", 2020),
        "doi": kw.get("doi", f"https://doi.org/10.1/{wid}"),
        "cited_by_count": kw.get("cited", 0),
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "type": kw.get("type", "article"),
        "is_retracted": kw.get("retracted", False),
        "authorships": kw.get("authorships", []),
        "primary_location": {"source": {"display_name": kw.get("venue", "Some Journal")}},
        "abstract_inverted_index": kw.get("abstract_index"),
    }


def test_parse_maps_core_fields(tmp_path) -> None:
    work = _work("W1", refs=("W2",), doi="https://doi.org/10.1038/x", venue="Nature",
                 type="review", retracted=True, cited=42)
    result = run_harvest(_FakeSource([work]), "q", cache_dir=tmp_path)
    p = result.papers[0]
    assert isinstance(p, HarvestedPaper)
    assert p.openalex_id == "W1"
    assert p.doi == "10.1038/x"  # https://doi.org/ prefix stripped
    assert p.venue == "Nature"
    assert p.is_review is True
    assert p.is_retracted is True
    assert p.cited_by_count == 42
    assert p.referenced_works == ("W2",)


def test_abstract_reconstructed_from_inverted_index(tmp_path) -> None:
    idx = {"Quantum": [0], "error": [1], "correction": [2]}
    result = run_harvest(_FakeSource([_work("W1", abstract_index=idx)]), "q", cache_dir=tmp_path)
    assert result.papers[0].abstract == "Quantum error correction"


def test_institution_countries_deduped_in_order(tmp_path) -> None:
    authorships = [
        {"institutions": [{"country_code": "US"}, {"country_code": "US"}]},
        {"institutions": [{"country_code": "DE"}]},
    ]
    source = _FakeSource([_work("W1", authorships=authorships)])
    result = run_harvest(source, "q", cache_dir=tmp_path)
    assert result.papers[0].institution_countries == ("US", "DE")


def test_in_subgraph_edges_keep_only_internal_citations(tmp_path) -> None:
    # W1 -> W2 (internal) and W1 -> WX (external, dropped); W3 -> W1 (internal).
    works = [
        _work("W1", refs=("W2", "WX")),
        _work("W2", refs=()),
        _work("W3", refs=("W1",)),
    ]
    result = run_harvest(_FakeSource(works), "q", cache_dir=tmp_path)
    assert set(result.edges) == {("W1", "W2"), ("W3", "W1")}
    assert ("W1", "WX") not in result.edges  # external target excluded


def test_self_citation_is_dropped_and_flagged(tmp_path) -> None:
    # W1's references include its own id (upstream data error) -> no self-loop,
    # but the id is surfaced so a recurrence is never silent.
    works = [_work("W1", refs=("W1", "W2")), _work("W2")]
    result = run_harvest(_FakeSource(works), "q", cache_dir=tmp_path)
    assert ("W1", "W1") not in result.edges
    assert result.edges == (("W1", "W2"),)
    assert result.self_referencing_ids == ("W1",)


def test_cache_prevents_refetch(tmp_path) -> None:
    source = _FakeSource([_work("W1"), _work("W2", refs=("W1",))])
    first = run_harvest(source, "quantum", cache_dir=tmp_path)
    second = run_harvest(source, "quantum", cache_dir=tmp_path)
    assert source.calls == 1  # second run served from cache
    assert first.edges == second.edges
    assert [p.openalex_id for p in first.papers] == [p.openalex_id for p in second.papers]


def test_use_cache_false_always_fetches(tmp_path) -> None:
    source = _FakeSource([_work("W1")])
    run_harvest(source, "q", cache_dir=tmp_path, use_cache=False)
    run_harvest(source, "q", cache_dir=tmp_path, use_cache=False)
    assert source.calls == 2


def test_expansion_pulls_external_works_and_cuts_leakage(tmp_path) -> None:
    # Seed = S1,S2 both citing external foundational F1 (+ S1 cites S2, internal).
    # F1 is external in the seed -> 2/3 of seed refs leak. Expansion should pull
    # F1 in, converting those to internal edges and dropping leakage.
    seed = [_work("S1", refs=("S2", "F1")), _work("S2", refs=("F1",))]
    universe = [*seed, _work("F1", refs=())]  # F1 fetchable by id
    source = _FakeSource(universe, seed=seed)

    no_expand = run_harvest(source, "q", cache_dir=tmp_path, use_cache=False)
    assert no_expand.seed_leakage > 0.0  # F1 leaks in the seed-only graph

    expanded = run_harvest(
        source, "q", cache_dir=tmp_path, use_cache=False, expand_rounds=2, expand_per_round=10
    )
    ids = {p.openalex_id for p in expanded.papers}
    assert "F1" in ids  # foundational work pulled into the set
    assert expanded.expansion_rounds >= 1
    # F1's incoming citations are now internal edges.
    assert ("S1", "F1") in expanded.edges and ("S2", "F1") in expanded.edges
    assert _leakage_of(expanded) < no_expand.seed_leakage


def test_expansion_off_by_default(tmp_path) -> None:
    seed = [_work("S1", refs=("F1",))]
    source = _FakeSource([*seed, _work("F1")], seed=seed)
    result = run_harvest(source, "q", cache_dir=tmp_path)
    assert result.expansion_rounds == 0
    assert source.fetch_calls == 0  # no expansion -> no id fetches
    assert {p.openalex_id for p in result.papers} == {"S1"}


def _leakage_of(result: object) -> float:
    from magnetor.robustness import boundary_leakage

    return boundary_leakage(result)  # type: ignore[arg-type]
