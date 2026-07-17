"""VoyageEmbedder: request shape, response parsing, batching, config guard."""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from magnetor import _http
from magnetor.embeddings.voyage import VoyageEmbedder
from magnetor.errors import ConfigError, ParseError


def _embedding_response(*vectors: list[float]) -> str:
    return json.dumps(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": v, "index": i}
                for i, v in enumerate(vectors)
            ],
            "model": "voyage-3.5",
            "usage": {"total_tokens": 10},
        }
    )


def _embedder(
    client: httpx.Client, *, dimension: int = 1024, batch_size: int = 1000
) -> VoyageEmbedder:
    return VoyageEmbedder(
        api_key="test-key",
        client=client,
        throttle=_http.NO_THROTTLE,
        dimension=dimension,
        batch_size=batch_size,
    )


def test_embed_documents_sets_input_type_and_orders_by_index(httpx_mock: HTTPXMock) -> None:
    # Deliberately return out of index order to prove we re-sort.
    httpx_mock.add_response(
        json=json.loads(
            json.dumps(
                {
                    "data": [
                        {"embedding": [0.0, 1.0], "index": 1},
                        {"embedding": [1.0, 0.0], "index": 0},
                    ]
                }
            )
        )
    )
    with httpx.Client() as client:
        embedder = _embedder(client, dimension=2)
        vectors = embedder.embed_documents(["alpha", "beta"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["input_type"] == "document"
    assert body["model"] == "voyage-4-lite"
    assert body["output_dimension"] == 2


def test_embed_query_uses_query_input_type(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_embedding_response([0.5, 0.5]))
    with httpx.Client() as client:
        vector = _embedder(client, dimension=2).embed_query("what is spin?")
    assert vector == [0.5, 0.5]
    assert json.loads(httpx_mock.get_requests()[0].content)["input_type"] == "query"


def test_authorization_header_is_sent(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_embedding_response([1.0]))
    with httpx.Client() as client:
        _embedder(client, dimension=1).embed_query("q")
    assert httpx_mock.get_requests()[0].headers["Authorization"] == "Bearer test-key"


def test_batching_splits_large_document_sets(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_embedding_response([1.0], [2.0]))
    httpx_mock.add_response(text=_embedding_response([3.0]))
    with httpx.Client() as client:
        embedder = _embedder(client, dimension=1, batch_size=2)
        vectors = embedder.embed_documents(["a", "b", "c"])
    assert vectors == [[1.0], [2.0], [3.0]]
    assert len(httpx_mock.get_requests()) == 2  # 2 + 1


def test_empty_documents_makes_no_request(httpx_mock: HTTPXMock) -> None:
    with httpx.Client() as client:
        assert _embedder(client).embed_documents([]) == []
    assert httpx_mock.get_requests() == []


def test_missing_api_key_raises_config_error() -> None:
    embedder = VoyageEmbedder(api_key=None)
    with pytest.raises(ConfigError):
        embedder.embed_query("q")


def test_count_mismatch_raises_parse_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=_embedding_response([1.0, 0.0]))  # only 1 back
    with httpx.Client() as client, pytest.raises(ParseError):
        _embedder(client, dimension=2).embed_documents(["a", "b"])
