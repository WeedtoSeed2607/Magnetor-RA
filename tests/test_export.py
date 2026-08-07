"""Branch C — packaging traced lineages as a downloadable citation bundle."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

from magnetor.export import (
    LineageSet,
    lineage_bundle,
    lineages_markdown,
    papers_bibtex,
    papers_csv,
    slugify,
    zip_bundle,
)


def _doc() -> dict[str, Any]:
    return {
        "query": "Defining Morality",
        "generated_at": "2026-08-06T00:00:00+00:00",
        "boundary_leakage": 0.92,
        "nodes": [
            {
                "id": "W1", "title": "The emotional dog & its rational tail",
                "year": 2001, "venue": "Psych Review", "doi": "10.1/abc",
                "url": "https://doi.org/10.1/abc", "influence": 1.0, "in_degree": 48,
                "pagerank": 0.033, "median_rank": 1, "lo_rank": 1, "hi_rank": 3,
                "stable": True, "is_retracted": False, "is_review": False,
            },
            {
                "id": "W2", "title": "Older work", "year": 1978, "venue": None,
                "doi": None, "url": "https://openalex.org/W2", "influence": 0.5,
                "in_degree": 3, "pagerank": 0.01, "is_retracted": True,
            },
            {"id": "W3", "title": "Unused paper", "year": 1990, "influence": 0.1},
        ],
        "edges": [["W1", "W2"]],
    }


def _sets() -> list[LineageSet]:
    return [LineageSet(label="W1 -> W2", kind="lineage", routes=(("W1", "W2"),))]


def test_slugify_makes_a_safe_folder_name() -> None:
    assert slugify("Defining Morality") == "defining-morality"
    assert slugify("!!!") == "evidence-graph"


def test_bundle_nests_everything_under_one_folder() -> None:
    files = lineage_bundle(_doc(), _sets())
    assert all(name.startswith("defining-morality/") for name in files)
    assert set(files) == {
        "defining-morality/README.md",
        "defining-morality/lineages.md",
        "defining-morality/papers.csv",
        "defining-morality/papers.bib",
        "defining-morality/lineages.json",
    }


def test_bundle_includes_only_papers_on_a_route() -> None:
    files = lineage_bundle(_doc(), _sets())
    assert "Unused paper" not in files["defining-morality/papers.csv"]
    assert "emotional dog" in files["defining-morality/papers.csv"]


def test_csv_has_a_row_per_paper_with_score_components() -> None:
    rows = list(csv.DictReader(io.StringIO(papers_csv(_doc()["nodes"][:2]))))
    assert len(rows) == 2
    assert rows[0]["openalex_id"] == "W1"
    assert rows[0]["in_set_citations"] == "48"
    assert rows[0]["rank_ci_high"] == "3"
    assert rows[1]["is_retracted"] == "True"


def test_bibtex_escapes_specials_and_keys_are_unique() -> None:
    bib = papers_bibtex(_doc()["nodes"][:2])
    assert r"\&" in bib  # the ampersand in the title is escaped
    assert bib.count("@article{") == 2
    keys = [line.split("{", 1)[1].rstrip(",") for line in bib.splitlines() if "@article{" in line]
    assert len(set(keys)) == 2


def test_bibtex_omits_author_rather_than_inventing_one() -> None:
    """Authors are not persisted on a graph node, so the field is absent."""
    assert "author" not in papers_bibtex(_doc()["nodes"][:1])


def test_bibtex_flags_a_retraction() -> None:
    assert "RETRACTED" in papers_bibtex([_doc()["nodes"][1]])


def test_markdown_lists_routes_oldest_first_with_links() -> None:
    md = lineages_markdown(_doc(), _sets())
    assert "1. (2001) [The emotional dog" in md
    assert "2. (1978) [Older work]" in md
    assert "**[RETRACTED]**" in md


def test_readme_carries_the_caveats_with_the_download() -> None:
    readme = lineage_bundle(_doc(), _sets())["defining-morality/README.md"]
    assert "Not the papers" in readme  # no PDFs, and it says why
    assert "92%" in readme  # boundary leakage travels with the file
    assert "rough guide" in readme
    assert "reprint" in readme


def test_json_payload_round_trips() -> None:
    payload = json.loads(lineage_bundle(_doc(), _sets())["defining-morality/lineages.json"])
    assert payload["query"] == "Defining Morality"
    assert payload["lineages"][0]["routes"] == [["W1", "W2"]]
    assert {p["id"] for p in payload["papers"]} == {"W1", "W2"}


def test_zip_contains_every_file_and_is_reproducible() -> None:
    files = lineage_bundle(_doc(), _sets())
    blob = zip_bundle(files)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert set(archive.namelist()) == set(files)
        assert archive.read("defining-morality/papers.bib").decode().startswith("@article{")
    assert blob == zip_bundle(files)  # no embedded clock


def test_empty_routes_produce_an_empty_paper_list() -> None:
    files = lineage_bundle(_doc(), [LineageSet(label="none", kind="none", routes=())])
    assert files["defining-morality/papers.bib"] == ""
    assert "No route found" in files["defining-morality/lineages.md"]
