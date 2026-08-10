"""Getting from a trend signal to the papers behind it."""

from __future__ import annotations

import pytest

from magnetor.dashboard_data import papers_for_term, papers_for_topic
from magnetor.resources import DomainStore
from magnetor.types import Domain

from .conftest import make_paper


@pytest.fixture
def stocked(tmp_path, monkeypatch):
    """A small qm corpus with known wording."""
    import magnetor.config as config

    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    store = DomainStore(Domain.QUANTUM_MECHANICS, tmp_path / "qm")
    store.store(make_paper(external_id="a", title="Entanglement in spin chains",
                           abstract="We measure entanglement entropy."))
    store.store(make_paper(external_id="b", title="Decoherence rates",
                           abstract="Entanglement is destroyed by decoherence."))
    store.store(make_paper(external_id="c", title="Topology of manifolds",
                           abstract="Purely geometric work."))
    return tmp_path


def test_a_term_finds_the_papers_using_it(stocked) -> None:
    found = papers_for_term(Domain.QUANTUM_MECHANICS, "entanglement")
    assert {p.external_id for p in found} == {"a", "b"}


def test_the_term_is_matched_in_title_or_abstract(stocked) -> None:
    """The trend engine counts both, so the search must reach both."""

    def ids(term: str) -> set[str]:
        return {p.external_id for p in papers_for_term(Domain.QUANTUM_MECHANICS, term)}

    assert "b" in ids("entanglement")  # only in the abstract
    assert "a" in ids("spin")  # only in the title


def test_matching_is_whole_word_not_substring(stocked) -> None:
    """"top" must not drag in "topology" — a signal nobody would recognise."""
    assert papers_for_term(Domain.QUANTUM_MECHANICS, "opolog") == ()


def test_an_absent_term_returns_nothing(stocked) -> None:
    assert papers_for_term(Domain.QUANTUM_MECHANICS, "cosmology") == ()


def test_an_empty_term_is_not_a_wildcard(stocked) -> None:
    assert papers_for_term(Domain.QUANTUM_MECHANICS, "   ") == ()


def test_the_limit_is_respected(stocked) -> None:
    assert len(papers_for_term(Domain.QUANTUM_MECHANICS, "entanglement", limit=1)) == 1


def test_a_topic_ranks_by_how_many_keywords_match(stocked) -> None:
    found = papers_for_topic(
        Domain.QUANTUM_MECHANICS, ["entanglement", "decoherence", "spin"]
    )
    assert found
    ids = [p.external_id for p, _hits in found]
    assert ids[0] in {"a", "b"}
    assert all(hits >= 2 for _p, hits in found)  # one shared word is not a theme


def test_a_topic_needs_more_than_one_keyword_hit(stocked) -> None:
    """Otherwise every paper joins the theme via its commonest word."""
    assert papers_for_topic(Domain.QUANTUM_MECHANICS, ["entanglement", "cosmology"]) == ()


def test_an_empty_topic_matches_nothing(stocked) -> None:
    assert papers_for_topic(Domain.QUANTUM_MECHANICS, []) == ()
    assert papers_for_topic(Domain.QUANTUM_MECHANICS, ["  ", ""]) == ()
