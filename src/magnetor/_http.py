"""Shared HTTP boundary for every outbound call.

Centralises timeout, politeness spacing, retry/backoff, and error-translation
policy so every caller — acquisition sources, the embedding client, and later
LLM clients — turns transport failures into Magnetor's own exception vocabulary
and stays a good citizen against rate-limited APIs (arXiv asks for ~3s between
requests; NCBI caps at 3 req/s without a key).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from magnetor.errors import SourceUnavailableError

#: Conservative default; callers are polite background clients, not
#: latency-sensitive foreground calls.
DEFAULT_TIMEOUT = httpx.Timeout(30.0)

#: Statuses worth retrying: explicit rate limiting plus transient server errors.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class Throttle:
    """Spaces successive requests by at least ``min_interval`` seconds.

    Stateful and single-threaded by design: one per client. The clock and sleep
    functions are injectable so tests neither wait nor depend on wall time.
    """

    def __init__(
        self,
        min_interval: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        """Block until at least ``min_interval`` has passed since the last call."""
        if self._min <= 0:
            return
        now = self._clock()
        if self._last is not None:
            remaining = self._min - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


#: Shared no-op throttle for callers (and tests) that want no spacing.
NO_THROTTLE = Throttle(0.0)


def get(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    throttle: Throttle | None = None,
    retries: int = 2,
    backoff_base: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """GET ``url`` and return the response text (see :func:`_send`)."""
    return _send(
        "GET",
        url,
        params=params,
        json_body=None,
        headers=headers,
        client=client,
        timeout=timeout,
        throttle=throttle,
        retries=retries,
        backoff_base=backoff_base,
        sleep=sleep,
    )


def post(
    url: str,
    *,
    json_body: object,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    throttle: Throttle | None = None,
    retries: int = 2,
    backoff_base: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """POST a JSON body to ``url`` and return the response text (see :func:`_send`)."""
    return _send(
        "POST",
        url,
        params=None,
        json_body=json_body,
        headers=headers,
        client=client,
        timeout=timeout,
        throttle=throttle,
        retries=retries,
        backoff_base=backoff_base,
        sleep=sleep,
    )


def _send(
    method: str,
    url: str,
    *,
    params: dict[str, str] | None,
    json_body: object,
    headers: dict[str, str] | None,
    client: httpx.Client | None,
    timeout: httpx.Timeout,
    throttle: Throttle | None,
    retries: int,
    backoff_base: float,
    sleep: Callable[[float], None],
) -> str:
    """Send one request, returning the response text.

    Applies the throttle before each attempt, retries transient failures
    (429/5xx) up to ``retries`` times honouring ``Retry-After`` when present,
    and translates every terminal failure into :class:`SourceUnavailableError`.
    """
    owns_client = client is None
    active = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(retries + 1):
            if throttle is not None:
                throttle.wait()
            try:
                response = active.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=True,
                )
            except httpx.HTTPError as exc:
                raise SourceUnavailableError(f"{url} unreachable: {exc}") from exc

            if response.status_code in _RETRY_STATUSES and attempt < retries:
                sleep(_retry_delay(response, attempt, backoff_base))
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise SourceUnavailableError(
                    f"{url} returned HTTP {exc.response.status_code}"
                ) from exc
            return response.text

        # Loop always returns or raises above; this guards the type checker.
        raise SourceUnavailableError(f"{url}: exhausted retries")
    finally:
        if owns_client:
            active.close()


def _retry_delay(response: httpx.Response, attempt: int, backoff_base: float) -> float:
    """Seconds to wait before the next attempt (Retry-After wins if numeric)."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return backoff_base * (2.0**attempt)
