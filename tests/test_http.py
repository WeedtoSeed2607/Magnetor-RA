"""HTTP boundary: retry/backoff, throttle, and error translation."""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from magnetor import _http
from magnetor.errors import SourceUnavailableError


def test_success_returns_body(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text="hello")
    with httpx.Client() as client:
        assert _http.get("https://x.test", client=client) == "hello"


def test_retries_then_succeeds(httpx_mock: HTTPXMock) -> None:
    slept: list[float] = []
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "0"})
    httpx_mock.add_response(text="ok")
    with httpx.Client() as client:
        result = _http.get("https://x.test", client=client, sleep=slept.append)
    assert result == "ok"
    assert len(slept) == 1  # backed off once before the retry


def test_exhausts_retries_and_raises(httpx_mock: HTTPXMock) -> None:
    slept: list[float] = []
    httpx_mock.add_response(status_code=503, is_reusable=True)
    with httpx.Client() as client, pytest.raises(SourceUnavailableError):
        _http.get("https://x.test", client=client, retries=2, sleep=slept.append)
    assert len(slept) == 2  # two retries attempted before giving up


def test_non_retryable_status_raises_immediately(httpx_mock: HTTPXMock) -> None:
    slept: list[float] = []
    httpx_mock.add_response(status_code=404)
    with httpx.Client() as client, pytest.raises(SourceUnavailableError):
        _http.get("https://x.test", client=client, sleep=slept.append)
    assert slept == []  # 404 is not retried


def test_transport_error_is_translated(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    with httpx.Client() as client, pytest.raises(SourceUnavailableError):
        _http.get("https://x.test", client=client)


def test_throttle_spaces_calls_using_injected_clock() -> None:
    now = [0.0]
    slept: list[float] = []
    throttle = _http.Throttle(
        5.0, sleep=lambda s: slept.append(s), clock=lambda: now[0]
    )
    throttle.wait()  # first call: no prior timestamp, no sleep
    throttle.wait()  # immediately after: must wait the full interval
    assert slept == [5.0]
