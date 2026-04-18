"""Verify the semantic-match path really rejects false hits and accepts near-dupes.

The real sentence-transformers + faiss stack is ~500 MB to install, so we
inject stand-ins that implement the same encode/search surface. The
false-hit guards (different model, different system prompt, low
similarity) are what we really want to exercise here — a semantic cache
that hands back the wrong answer is worse than no cache.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from arena.gateway.cache import SemanticCache
from arena.gateway.client import GatewayResponse


class StubEmbedder:
    """Deterministic 'embedder' that maps each input to a fixed vector.

    If `text` is a key in `vectors`, return that vector. Otherwise return
    a zero vector. Normalisation is on the caller; we hand back unit
    vectors where possible.
    """

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors
        self._dim = next(iter(vectors.values())).shape[0]

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        arr = np.stack(
            [self._vectors.get(t, np.zeros(self._dim, dtype="float32")) for t in texts]
        ).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = np.divide(arr, norms, out=np.zeros_like(arr), where=norms != 0)
        return arr


class StubIndex:
    """Mimics faiss.IndexFlatIP over tiny float32 arrays."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors: list[np.ndarray] = []

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add(self, vec: np.ndarray) -> None:
        for row in vec:
            self._vectors.append(row.astype("float32"))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return np.array([[-1.0]]), np.array([[-1]])
        stored = np.stack(self._vectors)
        scores = (stored @ query.T).reshape(-1)
        top = np.argsort(-scores)[:k]
        return scores[top].reshape(1, -1), top.reshape(1, -1)


def _install_stubs(cache: SemanticCache, vectors: dict[str, np.ndarray]) -> None:
    cache._embedder = StubEmbedder(vectors)
    cache._index = StubIndex(dim=next(iter(vectors.values())).shape[0])
    cache._enable_semantic = True


def _resp(content: str = "ok") -> GatewayResponse:
    return GatewayResponse(
        content=content, model="gpt-4o-mini", input_tokens=1, output_tokens=1,
        latency_ms=1, raw={},
    )


def test_semantic_hit_when_similar_enough() -> None:
    cache = SemanticCache(semantic_threshold=0.95)
    _install_stubs(
        cache,
        {
            "refund my subscription": np.array([1.0, 0.0], dtype="float32"),
            "please refund my subscription": np.array([0.99, 0.1], dtype="float32"),
        },
    )
    cache.put(
        [{"role": "system", "content": "triage"}, {"role": "user", "content": "refund my subscription"}],
        "gpt-4o-mini",
        0.0,
        _resp("high/billing"),
    )
    # A semantically similar but textually different user message hits the cache.
    hit = cache.get(
        [
            {"role": "system", "content": "triage"},
            {"role": "user", "content": "please refund my subscription"},
        ],
        "gpt-4o-mini",
        0.0,
    )
    assert hit is not None
    assert hit.content == "high/billing"


def test_semantic_miss_when_system_prompt_differs() -> None:
    cache = SemanticCache(semantic_threshold=0.90)
    _install_stubs(
        cache,
        {
            "refund my subscription": np.array([1.0, 0.0], dtype="float32"),
            "please refund": np.array([1.0, 0.0], dtype="float32"),  # identical embedding!
        },
    )
    cache.put(
        [
            {"role": "system", "content": "triage-v1"},
            {"role": "user", "content": "refund my subscription"},
        ],
        "gpt-4o-mini",
        0.0,
        _resp("v1 answer"),
    )
    # Same user-text embedding but different system prompt — must NOT hit.
    hit = cache.get(
        [
            {"role": "system", "content": "triage-v2"},
            {"role": "user", "content": "please refund"},
        ],
        "gpt-4o-mini",
        0.0,
    )
    assert hit is None


def test_semantic_miss_when_below_threshold() -> None:
    cache = SemanticCache(semantic_threshold=0.99)
    _install_stubs(
        cache,
        {
            "refund": np.array([1.0, 0.0], dtype="float32"),
            "dark mode": np.array([0.0, 1.0], dtype="float32"),
        },
    )
    cache.put(
        [{"role": "system", "content": "triage"}, {"role": "user", "content": "refund"}],
        "gpt-4o-mini",
        0.0,
        _resp(),
    )
    assert (
        cache.get(
            [{"role": "system", "content": "triage"}, {"role": "user", "content": "dark mode"}],
            "gpt-4o-mini",
            0.0,
        )
        is None
    )


def test_semantic_disabled_when_temperature_positive() -> None:
    cache = SemanticCache(semantic_threshold=0.5)
    _install_stubs(
        cache,
        {"q": np.array([1.0, 0.0], dtype="float32")},
    )
    cache.put(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        "gpt-4o-mini",
        0.0,
        _resp(),
    )
    # Same call at temperature > 0 must bypass both exact and semantic layers.
    assert (
        cache.get(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "gpt-4o-mini",
            0.7,
        )
        is None
    )


def test_stub_does_not_leak_across_models() -> None:
    cache = SemanticCache(semantic_threshold=0.5)
    _install_stubs(cache, {"q": np.array([1.0, 0.0], dtype="float32")})
    cache.put(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        "gpt-4o-mini",
        0.0,
        _resp(),
    )
    assert (
        cache.get(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "claude-haiku-4-5",
            0.0,
        )
        is None
    )
