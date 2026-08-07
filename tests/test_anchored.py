"""Branch C — anchored mode: a graph built outward from one paper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from magnetor.anchored import (
    AnchoredSource,
    AnchorError,
    anchor_label,
    extract_doi,
    extract_openalex_id,
    run_anchored_harvest,
)

# Real OpenAlex ids are "W" + digits, and the resolver enforces that shape, so the
# fixture uses realistic ids with readable aliases rather than mnemonic strings.
SEED = "W10001"
ROOT = "W10002"
DEEP = "W10003"
CITER = "W10004"
OTHER = "W10005"
_NAMES = {SEED: "Seed", ROOT: "Root", DEEP: "Deep", CITER: "Citer", OTHER: "Other"}


def _work(wid: str, *, refs: tuple[str, ...] = (), cited: int = 0, year: int = 2000,
          doi: str | None = None) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{wid}",
        "display_name": f"Paper {_NAMES[wid]}",
        "title": f"Paper {_NAMES[wid]}",
        "publication_year": year,
        "doi": doi,
        "cited_by_count": cited,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "type": "article",
        "is_retracted": False,
        "authorships": [],
    }


class FakeNeighbours:
    """In-memory OpenAlex stand-in. Records calls so API cost can be asserted."""

    def __init__(self, works: dict[str, dict[str, Any]], citers: dict[str, list[str]]) -> None:
        self.works = works
        self.citers = citers
        self.cited_by_calls: list[str] = []
        self.fetch_calls = 0

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        self.fetch_calls += 1
        return [self.works[i] for i in ids if i in self.works]

    def cited_by(self, work_id: str, *, limit: int) -> list[dict[str, Any]]:
        self.cited_by_calls.append(work_id)
        return [self.works[i] for i in self.citers.get(work_id, [])][:limit]

    def resolve(self, reference: str) -> dict[str, Any] | None:
        wid = extract_openalex_id(reference)
        if wid:
            return self.works.get(wid)
        doi = extract_doi(reference)
        if doi:
            return next((w for w in self.works.values() if w.get("doi") == doi), None)
        return None


def _fixture() -> FakeNeighbours:
    """ROOT <- SEED <- CITER, plus a second-hop ancestor and an unrelated work."""
    works = {
        SEED: _work(SEED, refs=(ROOT,), year=2000, doi="10.1037/seed"),
        ROOT: _work(ROOT, refs=(DEEP,), year=1980),
        DEEP: _work(DEEP, year=1960),
        CITER: _work(CITER, refs=(SEED,), year=2020, cited=99),
        OTHER: _work(OTHER, year=2015),
    }
    return FakeNeighbours(works, citers={SEED: [CITER], CITER: [OTHER]})


def test_extract_identifiers_from_bare_values_and_urls() -> None:
    assert extract_openalex_id("W2123791568") == "W2123791568"
    assert extract_openalex_id("https://openalex.org/W2123791568") == "W2123791568"
    assert extract_openalex_id("10.1/abc") is None
    assert extract_doi("10.1037/0033-295x.108.4.814") == "10.1037/0033-295x.108.4.814"
    assert extract_doi("https://doi.org/10.1037/abc") == "10.1037/abc"
    assert extract_doi("(10.1037/abc).") == "10.1037/abc"  # trailing punctuation stripped
    assert extract_doi("W2123791568") is None
    # A DOI prefix is "10." plus at least four digits, so this is not one.
    assert extract_doi("10.1/abc") is None


def _ids(works: list[dict[str, Any]]) -> set[str]:
    return {w["id"].rsplit("/", 1)[-1] for w in works}


def test_search_gathers_both_directions_around_the_seed() -> None:
    fake = _fixture()
    ids = _ids(AnchoredSource(fake).search(SEED, limit=50))
    assert SEED in ids
    assert CITER in ids  # forward: what built on it
    assert ROOT in ids and DEEP in ids  # backward: two hops of antecedents
    assert OTHER not in ids  # not in the neighbourhood at this fanout


def test_backward_hops_are_bounded() -> None:
    fake = _fixture()
    ids = _ids(AnchoredSource(fake, backward_hops=1).search(SEED, limit=50))
    assert ROOT in ids
    assert DEEP not in ids  # one hop only


def test_forward_costs_one_query_by_default() -> None:
    """Forward expansion is the expensive direction, so the default is seed-only."""
    fake = _fixture()
    AnchoredSource(fake).search(SEED, limit=50)
    assert fake.cited_by_calls == [SEED]


def test_forward_fanout_expands_the_most_cited_citers() -> None:
    fake = _fixture()
    found = AnchoredSource(fake, forward_fanout=1).search(SEED, limit=50)
    assert fake.cited_by_calls == [SEED, CITER]
    assert OTHER in _ids(found)


def test_limit_is_respected() -> None:
    fake = _fixture()
    assert len(AnchoredSource(fake).search(SEED, limit=2)) == 2


def test_unresolvable_seed_raises_with_guidance() -> None:
    fake = _fixture()
    with pytest.raises(AnchorError, match="OpenAlex id"):
        AnchoredSource(fake).search("not-a-paper", limit=10)


def test_seed_resolves_by_doi_too() -> None:
    fake = _fixture()
    found = AnchoredSource(fake).search("https://doi.org/10.1037/seed", limit=10)
    assert SEED in _ids(found)


def test_anchor_label_prefers_the_title() -> None:
    assert anchor_label({"display_name": "A Paper", "publication_year": 1999}, "W1") == (
        "Anchored: A Paper (1999)"
    )
    assert anchor_label(None, "W1") == "Anchored: W1"


def test_run_anchored_harvest_relabels_and_builds_edges(tmp_path) -> None:
    fake = _fixture()
    result = run_anchored_harvest(fake, SEED, limit=50, cache_dir=tmp_path)
    assert result.query == "Anchored: Paper Seed (2000)"
    ids = {p.openalex_id for p in result.papers}
    assert {SEED, ROOT, CITER} <= ids
    # The pipeline's own edge builder ran: citing -> cited, in-set only.
    assert (SEED, ROOT) in result.edges
    assert (CITER, SEED) in result.edges


def test_anchored_result_feeds_the_existing_pipeline(tmp_path) -> None:
    """The whole point of implementing this as a WorksSource."""
    from magnetor.graph import build_graph_document
    from magnetor.graph_scoring import score_graph
    from magnetor.relations import derive_relations
    from magnetor.robustness import bootstrap_rank_cis

    result = run_anchored_harvest(_fixture(), SEED, limit=50, cache_dir=tmp_path)
    doc = build_graph_document(
        result, score_graph(result), bootstrap_rank_cis(result, resamples=20),
        relations=derive_relations(result),
    )
    assert doc["query"].startswith("Anchored:")
    assert doc["nodes"]
    assert "biblio_coupled" in doc
