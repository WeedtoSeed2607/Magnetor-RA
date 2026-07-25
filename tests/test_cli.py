"""CLI surface: argument handling and dry-run, no network."""

from __future__ import annotations

from magnetor.cli import main
from magnetor.config import get_domain_config
from magnetor.resources import DomainStore
from magnetor.types import Domain
from tests.conftest import FakeEmbedder, make_paper


def _store_paper(**kwargs: object) -> None:
    config = get_domain_config(Domain.QUANTUM_MECHANICS)
    store = DomainStore(Domain.QUANTUM_MECHANICS, config.storage_dir)
    store.store(make_paper(**kwargs))  # type: ignore[arg-type]


def test_dry_run_all_reports_each_domain(capsys) -> None:
    code = main(["acquire", "all", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    for token in ("qm", "math", "neuro", "philosophy", "anthro", "history"):
        assert f"[{token}]" in out
    assert "dry-run" in out


def test_unknown_domain_is_usage_error(capsys) -> None:
    code = main(["acquire", "chemistry"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown domain" in err


def test_no_command_prints_help(capsys) -> None:
    code = main([])
    assert code == 2


def test_single_domain_dry_run(capsys) -> None:
    code = main(["acquire", "qm", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[qm]" in out
    assert "[math]" not in out


def test_show_empty_store(capsys) -> None:
    code = main(["show", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no records stored" in out


def test_show_lists_stored_records(capsys) -> None:
    _store_paper(external_id="2601.00001")
    code = main(["show", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[2601.00001]" in out
    assert "A Test Paper" in out
    assert "abstract:" in out


def test_show_truncates_unless_full(capsys) -> None:
    long_abstract = "word " * 100  # ~500 chars, exceeds the preview cap
    _store_paper(external_id="2601.00002", abstract=long_abstract)

    truncated = main(["show", "qm"])
    out = capsys.readouterr().out
    assert truncated == 0
    assert "..." in out

    main(["show", "qm", "--full"])
    full_out = capsys.readouterr().out
    assert full_out.count("word") > 50  # full abstract printed


def test_embed_without_key_errors(capsys) -> None:
    _store_paper(external_id="2601.00003", abstract="quantum spin")
    code = main(["embed", "qm"])
    err = capsys.readouterr().err
    assert code == 1
    assert "MAGNETOR_VOYAGE_API_KEY" in err


def test_embed_happy_path_with_fake_embedder(capsys, monkeypatch) -> None:
    monkeypatch.setattr("magnetor.cli._build_embedder", lambda: FakeEmbedder())
    _store_paper(external_id="2601.00004", abstract="quantum spin entanglement")
    code = main(["embed", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[qm] embedded=1" in out
    assert "index_size=1" in out


def _embed_qm(monkeypatch) -> None:
    monkeypatch.setattr("magnetor.cli._build_embedder", lambda: FakeEmbedder())
    _store_paper(external_id="q1", abstract="quantum spin entanglement measurement")
    _store_paper(external_id="q2", abstract="algebraic topology homology")
    main(["embed", "qm"])


def test_query_routes_and_returns_hits(capsys, monkeypatch) -> None:
    _embed_qm(monkeypatch)
    capsys.readouterr()  # discard embed output
    code = main(["query", "quantum spin entanglement", "--k", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "router:" in out
    assert "selected: qm" in out
    assert "[qm]" in out


def test_query_domain_override(capsys, monkeypatch) -> None:
    _embed_qm(monkeypatch)
    capsys.readouterr()
    code = main(["query", "anything", "--domain", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "router:" not in out  # routing skipped when domain forced
    assert "[qm]" in out


def test_query_unknown_domain_is_usage_error(capsys, monkeypatch) -> None:
    monkeypatch.setattr("magnetor.cli._build_embedder", lambda: FakeEmbedder())
    code = main(["query", "q", "--domain", "chemistry"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown domain" in err


def test_query_with_nothing_embedded(capsys, monkeypatch) -> None:
    monkeypatch.setattr("magnetor.cli._build_embedder", lambda: FakeEmbedder())
    code = main(["query", "anything"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no results" in out


def test_status_reports_counts(capsys, monkeypatch) -> None:
    _embed_qm(monkeypatch)
    capsys.readouterr()
    code = main(["status", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "domain" in out and "records" in out and "embedded" in out
    # qm has 2 records, both embedded.
    qm_line = next(line for line in out.splitlines() if line.startswith("qm"))
    assert "2" in qm_line


def test_status_all_domains(capsys) -> None:
    code = main(["status"])
    out = capsys.readouterr().out
    assert code == 0
    for token in ("qm", "math", "neuro", "philosophy", "anthro", "history"):
        assert any(line.startswith(token) for line in out.splitlines())


class _NoNetS2:
    """Semantic Scholar stand-in for CLI tests — never touches the network."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def expand(self, paper: object) -> tuple[list[object], list[object]]:
        return [], []


def _prep_deepdive(monkeypatch) -> None:
    _embed_qm(monkeypatch)
    monkeypatch.setattr("magnetor.cli.SemanticScholarClient", _NoNetS2)


def test_deepdive_runs(capsys, monkeypatch) -> None:
    _prep_deepdive(monkeypatch)
    capsys.readouterr()
    code = main(["deepdive", "quantum spin entanglement", "--domain", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[qm]" in out


def test_deepdive_payload_is_grounded(capsys, monkeypatch) -> None:
    _prep_deepdive(monkeypatch)
    capsys.readouterr()
    code = main(["deepdive", "quantum spin entanglement", "--domain", "qm", "--payload"])
    out = capsys.readouterr().out
    assert code == 0
    assert "SYNTHESIS SLOT" in out
    assert "[GROUNDED" in out


def test_deepdive_unknown_domain(capsys, monkeypatch) -> None:
    monkeypatch.setattr("magnetor.cli._build_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("magnetor.cli.SemanticScholarClient", _NoNetS2)
    code = main(["deepdive", "q", "--domain", "chemistry"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown domain" in err


def test_trends_volume_gate_skips(capsys) -> None:
    _store_paper(external_id="t1", abstract="quantum error correction")
    code = main(["trends", "qm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "skipped" in out and "volume gate" in out


def test_trends_force_runs(capsys) -> None:
    # Distinct vocab per doc so max_df pruning leaves usable terms.
    for i, text in enumerate([
        "quantum error correction surface codes",
        "graph coloring chromatic number theory",
        "protein folding molecular dynamics simulation",
        "neural network transformer attention mechanism",
    ]):
        _store_paper(external_id=f"t{i}", abstract=text)
    code = main(["trends", "qm", "--force", "--topics", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[qm]" in out and "topic" in out


def test_dashboard_command_builds_streamlit_argv() -> None:
    from magnetor.cli import _dashboard_command

    cmd = _dashboard_command(8080)
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4].endswith("dashboard.py")
    assert cmd[-2:] == ["--server.port", "8080"]


def test_dashboard_command_omits_port_when_none() -> None:
    from magnetor.cli import _dashboard_command

    cmd = _dashboard_command(None)
    assert "--server.port" not in cmd
    assert cmd[-1].endswith("dashboard.py")


def test_dashboard_run_reports_missing_streamlit(capsys, monkeypatch) -> None:
    def _boom(_cmd: list[str]) -> int:
        raise FileNotFoundError

    monkeypatch.setattr("magnetor.cli.subprocess.call", _boom)
    code = main(["dashboard"])
    err = capsys.readouterr().err
    assert code == 2
    assert "streamlit is not installed" in err


class _FakeWorks:
    """Offline WorksSource for the harvest CLI test — no OpenAlex call."""

    def _work(self, wid: str, refs: tuple[str, ...]) -> dict[str, object]:
        return {
            "id": f"https://openalex.org/{wid}",
            "title": f"Paper {wid}",
            "publication_year": 2020,
            "referenced_works": [f"https://openalex.org/{r}" for r in refs],
            "cited_by_count": 1,
        }

    def search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        return [self._work("A", ()), self._work("B", ("A",)), self._work("C", ("A", "B"))]

    def fetch_by_ids(self, ids: object) -> list[dict[str, object]]:
        return []  # CLI test uses --expand 0; no expansion fetch expected


def test_harvest_builds_and_saves_graph(capsys, monkeypatch) -> None:
    monkeypatch.setattr("magnetor.cli._build_works_source", lambda: _FakeWorks())
    code = main(["harvest", "quantum error correction", "--resamples", "30"])
    out = capsys.readouterr().out
    assert code == 0
    assert "3 papers" in out and "saved ->" in out

    from magnetor.graph import load_graph
    doc = load_graph("quantum error correction")
    assert doc is not None
    assert doc["nodes"][0]["id"] == "A"  # most in-set-cited ranks first
