"""Voyage AI embedding adapter (Spec Section 7.1).

Calls Voyage's ``/v1/embeddings`` endpoint through the shared HTTP boundary,
setting ``input_type`` to ``document`` or ``query`` for the asymmetric split.
Configuration (key, model, dimension) comes from the environment so the same
code runs in any deployment; nothing is hard-coded.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence

import httpx

from magnetor import _http
from magnetor.errors import ConfigError, ParseError

_API_URL = "https://api.voyageai.com/v1/embeddings"

_API_KEY_ENV = "MAGNETOR_VOYAGE_API_KEY"
_MODEL_ENV = "MAGNETOR_VOYAGE_MODEL"
#: voyage-4 family carries the 200M-token free tier; -lite is the cheapest of
#: it. Override via MAGNETOR_VOYAGE_MODEL.
_DEFAULT_MODEL = "voyage-4-lite"
_DEFAULT_DIMENSION = 1024
#: Voyage accepts up to 1000 inputs per request.
_MAX_BATCH = 1000
#: Modest spacing; batching already keeps request counts low.
_REQUEST_INTERVAL = 0.2
#: Embedding is a background batch job, so ride out rate-limit windows patiently:
#: more retries and a longer backoff than the interactive default (429s are
#: common on Voyage's trial tier until a payment method lifts the limits).
_RETRIES = 4
_BACKOFF_BASE = 1.0


class VoyageEmbedder:
    """Embed text via the Voyage AI API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int = _DEFAULT_DIMENSION,
        batch_size: int = _MAX_BATCH,
        client: httpx.Client | None = None,
        throttle: _http.Throttle | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get(_API_KEY_ENV)
        self._model = model or os.environ.get(_MODEL_ENV, _DEFAULT_MODEL)
        self._dimension = dimension
        self._batch_size = max(1, min(batch_size, _MAX_BATCH))
        self._client = client
        self._throttle = throttle or _http.Throttle(_REQUEST_INTERVAL)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts), input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise ConfigError(
                f"{_API_KEY_ENV} is not set; cannot call the Voyage embedding API"
            )
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            body = {
                "input": batch,
                "model": self._model,
                "input_type": input_type,
                "output_dimension": self._dimension,
            }
            text = _http.post(
                _API_URL,
                json_body=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                client=self._client,
                throttle=self._throttle,
                retries=_RETRIES,
                backoff_base=_BACKOFF_BASE,
            )
            vectors.extend(_parse_embeddings(text, expected=len(batch)))
        return vectors


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _parse_embeddings(text: str, *, expected: int) -> list[list[float]]:
    """Extract embedding vectors from a Voyage response, ordered by ``index``."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Voyage returned non-JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParseError("Voyage payload was not an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ParseError("Voyage payload missing a data array")
    by_index: dict[int, list[float]] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ParseError("Voyage data entry was not an object")
        index = item.get("index")
        embedding = item.get("embedding")
        if not isinstance(index, int) or not isinstance(embedding, list):
            raise ParseError("Voyage data entry missing index/embedding")
        by_index[index] = [float(x) for x in embedding]
    if len(by_index) != expected:
        raise ParseError(
            f"Voyage returned {len(by_index)} embeddings for {expected} inputs"
        )
    return [by_index[i] for i in range(expected)]
