"""The exception hierarchy is a public contract; pin it down."""

from __future__ import annotations

from magnetor.errors import (
    AcquisitionError,
    ConfigError,
    MagnetorError,
    ParseError,
    RedistributionError,
    SourceUnavailableError,
)


def test_all_errors_derive_from_root() -> None:
    for err in (
        ConfigError,
        AcquisitionError,
        SourceUnavailableError,
        ParseError,
        RedistributionError,
    ):
        assert issubclass(err, MagnetorError)


def test_source_and_parse_errors_are_acquisition_errors() -> None:
    assert issubclass(SourceUnavailableError, AcquisitionError)
    assert issubclass(ParseError, AcquisitionError)


def test_catching_root_catches_specific() -> None:
    try:
        raise SourceUnavailableError("boom")
    except MagnetorError as exc:
        assert isinstance(exc, SourceUnavailableError)
