"""Project root exception and specific subclasses.

Follows the "one project root exception plus specific subclasses" baseline: any
error originating in Magnetor is a :class:`MagnetorError`, so callers can catch
the whole surface with a single ``except`` while still discriminating specific
failure modes when they need to.
"""

from __future__ import annotations


class MagnetorError(Exception):
    """Root of every exception Magnetor raises deliberately."""


class ConfigError(MagnetorError):
    """A domain/config value is missing, unknown, or internally inconsistent."""


class AcquisitionError(MagnetorError):
    """An acquisition run failed for a recoverable, source-specific reason."""


class SourceUnavailableError(AcquisitionError):
    """The upstream source could not be reached or returned an error status."""


class ParseError(AcquisitionError):
    """A source response was reached but could not be parsed into papers."""


class RedistributionError(MagnetorError):
    """Storing this content would violate the source's redistribution terms.

    Raised by the slow-group (batch/manual) acquisition path when full text is
    encountered without a license permitting local storage (Spec Section 4:
    "confirm permitted use with source before storing").
    """
