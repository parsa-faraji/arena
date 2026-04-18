from __future__ import annotations

from arena.gateway.cache import SemanticCache
from arena.gateway.client import GatewayResponse


def _make_response(content: str = "ok") -> GatewayResponse:
    return GatewayResponse(
        content=content,
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=2,
        latency_ms=42,
        raw={},
    )


def test_exact_hit() -> None:
    cache = SemanticCache(enable_semantic=False)
    messages = [{"role": "user", "content": "hello"}]
    cache.put(messages, "gpt-4o-mini", 0.0, _make_response("hi"))
    got = cache.get(messages, "gpt-4o-mini", 0.0)
    assert got is not None
    assert got.content == "hi"


def test_exact_miss_on_different_model() -> None:
    cache = SemanticCache(enable_semantic=False)
    messages = [{"role": "user", "content": "hello"}]
    cache.put(messages, "gpt-4o-mini", 0.0, _make_response("hi"))
    assert cache.get(messages, "claude-haiku-4-5", 0.0) is None


def test_no_cache_when_temperature_positive() -> None:
    cache = SemanticCache(enable_semantic=True)
    messages = [{"role": "user", "content": "hello"}]
    cache.put(messages, "gpt-4o-mini", 0.7, _make_response("hi"))
    # exact key differs by temperature so same call at 0.0 misses
    assert cache.get(messages, "gpt-4o-mini", 0.0) is None
