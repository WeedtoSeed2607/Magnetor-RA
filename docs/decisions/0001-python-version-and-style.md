# 0001 — Python version and style baseline

Status: Accepted · Date: 2026-07-04

## Context

Phase 1 needs a pinned runtime and an enforced style/typing baseline so the
codebase stays consistent as later phases (router, trend engine, Critic,
pedagogical layer) are added by potentially different hands.

## Decision

- **Runtime:** pin **CPython 3.14** (`requires-python = ">=3.14"`). This is the
  interpreter the platform owner installed (3.14.6); we build against it rather
  than an older target so we can use `StrEnum`, `datetime.UTC`, and modern
  typing syntax without shims.
- **Formatting + linting:** **Ruff** as the single tool (rule sets
  `E, F, I, UP, B, SIM, RUF`), line length 100. No manual PEP 8 discipline.
- **Type checking:** **mypy `strict`** over `src` and `tests`. Source is fully
  strict; the `tests.*` package relaxes only `disallow_untyped_defs` /
  `disallow_incomplete_defs` because pytest injects fixture parameters untyped.
- **Layout:** `src/` layout with the package under `src/magnetor`, tests under
  `tests/`, so an editable install exercises the same import paths as a wheel.

## Consequences

- New code must pass `ruff check`, `mypy`, and `pytest` before it is considered
  done. These three form the Phase 1 "green gate."
- Pinning 3.14 means dependencies must ship 3.14 wheels; verified at setup for
  httpx, pytest, pytest-httpx, ruff, and mypy.
- Boundary code that consumes untyped JSON (PubMed E-utilities) narrows values
  explicitly (`_expect_dict`, `isinstance` guards) rather than returning `Any`,
  keeping strict mode honest at the one place external data enters.
