"""Magnetor — Domain-Aware Research Platform.

Phase 1 implements the Acquisition layer described in the Revised Architecture
Specification v2.1 (Section 4). It pulls papers per domain into physically
isolated per-domain storage, honouring each source's real cadence and
redistribution terms.

Storage isolation is the load-bearing invariant: no document body or embedding
is ever copied across domain directories (Spec Section 3, "Rule (revised)").
"""

from magnetor.errors import MagnetorError

__all__ = ["MagnetorError", "__version__"]

__version__ = "0.1.0"
