"""Semantic cache for gateway calls.

Two layers:
  1. exact-key cache (SHA of messages+model+temperature) — always on, cheap.
  2. semantic cache — embeds the user message, looks up nearest neighbour in
     a local FAISS index; if cosine similarity ≥ threshold AND the surrounding
     request metadata matches, returns the cached response.

The semantic layer is optional — if `sentence-transformers` / `faiss-cpu`
are not installed we fall back to exact-only. This keeps `pip install
arena` fast for users who don't need it.

False-hit guard: we never return a semantic hit across different models,
different temperatures > 0, or different system prompts. Semantic match is
only on the last user-role message.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.gateway.client import GatewayResponse

log = logging.getLogger(__name__)


def _hash_key(messages: list[dict[str, Any]], model: str, temperature: float) -> str:
    payload = json.dumps(
        {"m": messages, "model": model, "t": temperature},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CacheEntry:
    response: GatewayResponse
    signature: dict[str, Any]


class SemanticCache:
    """Hybrid exact + semantic cache.

    Thread-safe for concurrent variant runners. The embedding model is
    loaded lazily on first semantic lookup so importing this module stays
    cheap.
    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        semantic_threshold: float = 0.92,
        enable_semantic: bool = True,
    ) -> None:
        self._exact: dict[str, CacheEntry] = {}
        self._semantic_threshold = semantic_threshold
        self._enable_semantic = enable_semantic
        self._lock = threading.RLock()
        self._path = path

        self._embedder: Any | None = None
        self._index: Any | None = None
        self._index_entries: list[CacheEntry] = []

    # ------------------------------------------------------------------ API

    def get(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
    ) -> GatewayResponse | None:
        exact = _hash_key(messages, model, temperature)
        with self._lock:
            entry = self._exact.get(exact)
            if entry is not None:
                return entry.response

        if not self._enable_semantic or temperature > 0.0:
            return None

        user_text = _last_user_text(messages)
        if not user_text:
            return None

        signature = _signature(messages, model, temperature)
        return self._semantic_lookup(user_text, signature)

    def put(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        response: GatewayResponse,
    ) -> None:
        exact = _hash_key(messages, model, temperature)
        signature = _signature(messages, model, temperature)
        entry = CacheEntry(response=response, signature=signature)
        with self._lock:
            self._exact[exact] = entry
            if self._enable_semantic and temperature == 0.0:
                self._semantic_put(_last_user_text(messages), entry)

    # ---------------------------------------------------------- semantic

    def _ensure_embedder(self) -> bool:
        if self._embedder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except Exception as exc:
            log.info("semantic cache disabled: %s", exc)
            self._enable_semantic = False
            return False

    def _ensure_index(self, dim: int) -> bool:
        if self._index is not None:
            return True
        try:
            import faiss  # type: ignore

            self._index = faiss.IndexFlatIP(dim)
            return True
        except Exception as exc:
            log.info("faiss unavailable: %s", exc)
            self._enable_semantic = False
            return False

    def _semantic_put(self, text: str, entry: CacheEntry) -> None:
        if not text or not self._ensure_embedder():
            return
        import numpy as np

        vec = self._embedder.encode([text], normalize_embeddings=True)  # type: ignore[union-attr]
        vec = np.asarray(vec, dtype="float32")
        if not self._ensure_index(vec.shape[1]):
            return
        with self._lock:
            self._index.add(vec)  # type: ignore[union-attr]
            self._index_entries.append(entry)

    def _semantic_lookup(self, text: str, signature: dict[str, Any]) -> GatewayResponse | None:
        if not self._ensure_embedder() or self._index is None:
            return None
        import numpy as np

        vec = self._embedder.encode([text], normalize_embeddings=True)  # type: ignore[union-attr]
        vec = np.asarray(vec, dtype="float32")
        with self._lock:
            if self._index.ntotal == 0:  # type: ignore[union-attr]
                return None
            scores, idx = self._index.search(vec, 1)  # type: ignore[union-attr]
            top_score = float(scores[0][0])
            top_idx = int(idx[0][0])
            if top_idx < 0 or top_idx >= len(self._index_entries):
                return None
            candidate = self._index_entries[top_idx]

        if top_score < self._semantic_threshold:
            return None
        if candidate.signature != signature:
            return None
        log.debug("semantic cache hit: score=%.3f", top_score)
        return candidate.response

    def size(self) -> int:
        with self._lock:
            return len(self._exact)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal parts
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _signature(messages: list[dict[str, Any]], model: str, temperature: float) -> dict[str, Any]:
    system = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"),
        "",
    )
    return {"model": model, "temperature": temperature, "system": system}
