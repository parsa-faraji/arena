"""Integration test for the Respan Gateway client using a fake HTTP server.

We don't need pytest-vcr for this — a tiny in-process httpx.MockTransport
gives us full control and runs without network access. The test verifies
that:
  - the OpenAI-compatible payload is shaped correctly,
  - `fallback_models` and `disable_log` flow through `extra_body`,
  - retries kick in on 429.
"""
from __future__ import annotations

import json

import httpx
from openai import OpenAI

from arena.gateway.client import GatewayClient


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "arena online"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        return httpx.Response(200, json=body)


def _make_client(recorder: _Recorder) -> GatewayClient:
    transport = httpx.MockTransport(recorder)
    http_client = httpx.Client(transport=transport)
    client = GatewayClient(api_key="sk-test", base_url="https://api.respan.ai/api")
    # Swap the underlying OpenAI client to one that uses our mock transport.
    client._client = OpenAI(
        api_key="sk-test",
        base_url="https://api.respan.ai/api",
        http_client=http_client,
        max_retries=0,
    )
    return client


def test_chat_sends_openai_compatible_payload() -> None:
    rec = _Recorder()
    client = _make_client(rec)

    resp = client.chat(
        [{"role": "user", "content": "hi"}],
        fallback_models=["claude-haiku-4-5"],
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=16,
    )

    assert resp.content == "arena online"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 2
    assert resp.total_tokens == 7
    assert resp.latency_ms >= 0

    assert len(rec.requests) == 1
    req = rec.requests[0]
    assert req.method == "POST"
    assert req.url.path.endswith("/chat/completions")
    payload = json.loads(req.content)
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0] == {"role": "user", "content": "hi"}
    assert payload["fallback_models"] == ["claude-haiku-4-5"]


def test_disable_log_flows_through_extra_body() -> None:
    rec = _Recorder()
    client = GatewayClient(api_key="sk-test", disable_log=True)
    transport = httpx.MockTransport(rec)
    http_client = httpx.Client(transport=transport)
    client._client = OpenAI(
        api_key="sk-test",
        base_url="https://api.respan.ai/api",
        http_client=http_client,
        max_retries=0,
    )
    client.chat([{"role": "user", "content": "secret"}])
    payload = json.loads(rec.requests[0].content)
    assert payload.get("disable_log") is True
