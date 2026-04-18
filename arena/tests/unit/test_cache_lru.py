"""Exact-cache eviction must stay bounded and follow LRU order.

Unbounded caches are fine until the process runs for a day. A 200k-case
sweep with a dynamic prompt template grows `_exact` forever without this
cap.
"""
from __future__ import annotations

import pytest

from arena.gateway.cache import SemanticCache
from arena.gateway.client import GatewayResponse


def _resp(content: str = "r") -> GatewayResponse:
    return GatewayResponse(
        content=content, model="gpt-4o-mini", input_tokens=1, output_tokens=1,
        latency_ms=1, raw={},
    )


def test_cache_bounded_by_max_entries() -> None:
    cache = SemanticCache(enable_semantic=False, max_entries=3)
    for i in range(5):
        cache.put([{"role": "user", "content": f"q{i}"}], "gpt-4o-mini", 0.0, _resp(f"r{i}"))
    assert cache.size() == 3


def test_cache_evicts_least_recently_used() -> None:
    cache = SemanticCache(enable_semantic=False, max_entries=2)
    cache.put([{"role": "user", "content": "a"}], "gpt-4o-mini", 0.0, _resp("ra"))
    cache.put([{"role": "user", "content": "b"}], "gpt-4o-mini", 0.0, _resp("rb"))
    # Touch 'a' — it becomes most recently used.
    cache.get([{"role": "user", "content": "a"}], "gpt-4o-mini", 0.0)
    # Adding 'c' should evict 'b', not 'a'.
    cache.put([{"role": "user", "content": "c"}], "gpt-4o-mini", 0.0, _resp("rc"))
    assert cache.get([{"role": "user", "content": "a"}], "gpt-4o-mini", 0.0) is not None
    assert cache.get([{"role": "user", "content": "b"}], "gpt-4o-mini", 0.0) is None
    assert cache.get([{"role": "user", "content": "c"}], "gpt-4o-mini", 0.0) is not None


def test_max_entries_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        SemanticCache(max_entries=0)
