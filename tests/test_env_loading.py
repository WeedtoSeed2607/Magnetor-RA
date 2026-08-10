"""Loading the project's .env in process, so any entry point finds the API key."""

from __future__ import annotations

import os

from magnetor.config import load_env_file


def test_values_are_loaded_into_the_environment(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("MAGNETOR_TEST_KEY=abc123\n", encoding="utf-8")
    monkeypatch.delenv("MAGNETOR_TEST_KEY", raising=False)
    assert load_env_file(env) == ["MAGNETOR_TEST_KEY"]
    assert os.environ["MAGNETOR_TEST_KEY"] == "abc123"


def test_the_real_environment_always_wins(tmp_path, monkeypatch) -> None:
    """A container secret or Streamlit's own store must never be overridden."""
    env = tmp_path / ".env"
    env.write_text("MAGNETOR_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MAGNETOR_TEST_KEY", "from-environment")
    assert load_env_file(env) == []
    assert os.environ["MAGNETOR_TEST_KEY"] == "from-environment"


def test_comments_blanks_and_junk_are_skipped(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nMAGNETOR_TEST_KEY=ok\nnot-a-pair\nMAGNETOR_TEST_EMPTY=\n",
        encoding="utf-8",
    )
    for name in ("MAGNETOR_TEST_KEY", "MAGNETOR_TEST_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    assert load_env_file(env) == ["MAGNETOR_TEST_KEY"]
    assert "MAGNETOR_TEST_EMPTY" not in os.environ  # blank value sets nothing


def test_surrounding_quotes_are_stripped(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('MAGNETOR_TEST_KEY="quoted-value"\n', encoding="utf-8")
    monkeypatch.delenv("MAGNETOR_TEST_KEY", raising=False)
    load_env_file(env)
    assert os.environ["MAGNETOR_TEST_KEY"] == "quoted-value"


def test_a_missing_file_is_normal(tmp_path) -> None:
    """Hosted deployments have no .env and must not fail because of it."""
    assert load_env_file(tmp_path / "nope.env") == []


def test_returns_names_never_values(tmp_path, monkeypatch) -> None:
    """The return value is for logging, so a secret must not travel in it."""
    env = tmp_path / ".env"
    env.write_text("MAGNETOR_TEST_KEY=super-secret\n", encoding="utf-8")
    monkeypatch.delenv("MAGNETOR_TEST_KEY", raising=False)
    assert "super-secret" not in "".join(load_env_file(env))
