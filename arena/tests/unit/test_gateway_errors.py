"""All transport failures from the OpenAI client must surface as GatewayError.

If these leak out as raw openai exceptions, the runner's `except GatewayError`
misses them and a single network blip crashes a 200-case eval. That regression
would be silent in unit tests but obvious on first real use.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import OpenAI

from arena.gateway.client import GatewayClient, GatewayError


def _always_fails(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("simulated connection refused")


def _rate_limits(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, json={"error": {"message": "rate limited"}})


def _server_errors(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": {"message": "service unavailable"}})


def _client_with(handler: Any) -> GatewayClient:
    client = GatewayClient(api_key="sk-test")
    client._client = OpenAI(
        api_key="sk-test",
        base_url="https://api.respan.ai/api",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return client


def test_connection_error_wrapped() -> None:
    client = _client_with(_always_fails)
    with pytest.raises(GatewayError, match="request failed|unreachable|timed out"):
        client.chat([{"role": "user", "content": "hi"}])


def test_rate_limit_after_retries_wrapped() -> None:
    client = _client_with(_rate_limits)
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "hi"}])


def test_server_error_wrapped() -> None:
    client = _client_with(_server_errors)
    with pytest.raises(GatewayError, match="503"):
        client.chat([{"role": "user", "content": "hi"}])
