from __future__ import annotations

from typing import Any

import pytest

from arena.gateway.client import GatewayError, GatewayResponse
from arena.gateway.fallback import FallbackChain


def _resp(model: str) -> GatewayResponse:
    return GatewayResponse(
        content="ok",
        model=model,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        raw={},
    )


def test_first_link_succeeds() -> None:
    chain = FallbackChain([("a", []), ("b", [])])

    def chat_fn(**kwargs: Any) -> GatewayResponse:
        return _resp(kwargs["model"])

    r = chain.run(chat_fn, messages=[], temperature=0.0)
    assert r.model == "a"


def test_second_link_used_when_first_fails() -> None:
    chain = FallbackChain([("a", []), ("b", [])])
    calls: list[str] = []

    def chat_fn(**kwargs: Any) -> GatewayResponse:
        calls.append(kwargs["model"])
        if kwargs["model"] == "a":
            raise GatewayError("boom")
        return _resp(kwargs["model"])

    r = chain.run(chat_fn, messages=[], temperature=0.0)
    assert r.model == "b"
    assert calls == ["a", "b"]


def test_all_fail_raises() -> None:
    chain = FallbackChain([("a", []), ("b", [])])

    def chat_fn(**kwargs: Any) -> GatewayResponse:
        raise GatewayError("always")

    with pytest.raises(GatewayError, match="exhausted"):
        chain.run(chat_fn, messages=[], temperature=0.0)


def test_server_fallbacks_forwarded() -> None:
    chain = FallbackChain([("a", ["a-backup"])])
    seen: dict[str, Any] = {}

    def chat_fn(**kwargs: Any) -> GatewayResponse:
        seen.update(kwargs)
        return _resp(kwargs["model"])

    chain.run(chat_fn, messages=[])
    assert seen["fallback_models"] == ("a-backup",)
