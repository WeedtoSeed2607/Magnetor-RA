"""Config registry and domain resolution."""

from __future__ import annotations

import datetime as dt

import pytest

from magnetor.config import all_domains, get_domain_config, resolve_domain
from magnetor.errors import ConfigError
from magnetor.types import AcquisitionMode, Domain


def test_every_domain_has_a_config() -> None:
    for domain in Domain:
        config = get_domain_config(domain)
        assert config.domain is domain
        assert config.trend_model
        assert isinstance(config.cadence, dt.timedelta)


def test_fast_group_is_automated_daily() -> None:
    for domain in (Domain.QUANTUM_MECHANICS, Domain.MATHEMATICS, Domain.NEUROSCIENCE):
        config = get_domain_config(domain)
        assert config.acquisition_mode is AcquisitionMode.AUTOMATED_BULK
        assert config.cadence == dt.timedelta(days=1)


def test_slow_group_is_batch_manual() -> None:
    for domain in (Domain.PHILOSOPHY, Domain.ANTHROPOLOGY, Domain.HISTORY):
        config = get_domain_config(domain)
        assert config.acquisition_mode is AcquisitionMode.BATCH_MANUAL
        assert config.cadence >= dt.timedelta(days=7)


def test_storage_dir_is_isolated_per_domain() -> None:
    qm = get_domain_config(Domain.QUANTUM_MECHANICS).storage_dir
    math = get_domain_config(Domain.MATHEMATICS).storage_dir
    assert qm != math
    assert qm.name == "qm"
    assert math.name == "math"


def test_resolve_domain_maps_tokens() -> None:
    assert resolve_domain("qm") is Domain.QUANTUM_MECHANICS
    assert resolve_domain("neuro") is Domain.NEUROSCIENCE


def test_resolve_domain_rejects_unknown() -> None:
    with pytest.raises(ConfigError):
        resolve_domain("chemistry")


def test_all_domains_has_six() -> None:
    assert len(all_domains()) == 6
