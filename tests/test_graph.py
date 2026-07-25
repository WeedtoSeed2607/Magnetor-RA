"""Branch C · L4 graph: document assembly, persistence, listing."""

from __future__ import annotations

import datetime as dt
from typing import Any

from magnetor.graph import (
    build_graph_document,
    list_graphs,
    load_graph,
    query_hash,
    save_graph,
)
from magnetor.graph_scoring import score_graph
from magnetor.harvest import HarvestedPaper, HarvestResult
from magnetor.robustness import bootstrap_rank_cis


def _paper(wid: str, refs: tuple[str, ...] = (), *, retracted: bool = False,
           doi: str | None = None) -> HarvestedPaper:
    return HarvestedPaper(
        openalex_id=wid, title=f"Paper {wid}", year=2020, doi=doi, venue="V",
        cited_by_count=0, referenced_works=refs, institution_countries=(),
        is_review=False, is_retracted=retracted,
    )


def _result() -> HarvestResult:
    papers = [
        _paper("A", doi="10.1/a"),
        _paper("B", ("A",)),
        _paper("C", ("A", "B"), retracted=True),
    ]
    return HarvestResult(
        query="quantum error correction",
        generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(),
        papers=tuple(papers), edges=(("B", "A"), ("C", "A"), ("C", "B")), n_fetched=3,
    )


def _doc(top_n: int | None = None) -> dict[str, Any]:
    res = _result()
    robustness = bootstrap_rank_cis(res, resamples=40)
    return build_graph_document(res, score_graph(res), robustness, top_n=top_n)


def test_build_joins_scores_flags_and_links() -> None:
    doc = _doc()
    by_id = {n["id"]: n for n in doc["nodes"]}
    assert doc["nodes"][0]["id"] == "A"  # most in-set-cited -> first
    assert by_id["A"]["in_degree"] == 2
    assert by_id["A"]["url"] == "https://doi.org/10.1/a"
    assert by_id["C"]["is_retracted"] is True
    assert by_id["B"]["url"] == "https://openalex.org/B"  # no DOI -> OpenAlex URL
    assert doc["boundary_leakage"] == 0.0  # every ref is in-set here


def test_top_n_limits_nodes_and_prunes_edges() -> None:
    doc = _doc(top_n=2)
    assert len(doc["nodes"]) == 2
    kept = {n["id"] for n in doc["nodes"]}
    for u, v in doc["edges"]:
        assert u in kept and v in kept


def test_save_load_roundtrip(tmp_path) -> None:
    doc = _doc()
    path = save_graph(doc, graphs_dir=tmp_path)
    assert path.exists()
    loaded = load_graph("quantum error correction", graphs_dir=tmp_path)
    assert loaded is not None
    assert loaded["query"] == "quantum error correction"


def test_query_hash_is_case_and_space_insensitive() -> None:
    assert query_hash("Quantum Error Correction ") == query_hash("quantum error correction")


def test_load_missing_returns_none(tmp_path) -> None:
    assert load_graph("never harvested", graphs_dir=tmp_path) is None


def test_list_graphs_summarises_saved(tmp_path) -> None:
    save_graph(_doc(), graphs_dir=tmp_path)
    listed = list_graphs(graphs_dir=tmp_path)
    assert len(listed) == 1
    assert listed[0]["query"] == "quantum error correction"
    assert listed[0]["n_nodes"] == 3
