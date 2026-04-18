"""Client-side fallback chain.

Respan Gateway already supports server-side `fallback_models`. This module
is the belt-and-suspenders layer: if the gateway itself errors (network,
5xx, all server fallbacks exhausted), we try the next *chain* — typically
a different model family — with fresh exponential backoff.

Usage:
    chain = FallbackChain([
        ("gpt-4o-mini", ["gpt-4o"]),
        ("claude-haiku-4-5", ["claude-sonnet-4-6"]),
    ])
    resp = chain.run(client.chat, messages=..., temperature=0)
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from arena.gateway.client import GatewayError, GatewayResponse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChainLink:
    primary: str
    fallbacks: Sequence[str] = ()


class FallbackChain:
    """Ordered list of primary/fallback pairs; tries each link until one succeeds."""

    def __init__(self, links: Sequence[tuple[str, Sequence[str]] | ChainLink]) -> None:
        self._links: list[ChainLink] = [
            ChainLink(primary=link[0], fallbacks=tuple(link[1]))
            if isinstance(link, tuple)
            else link
            for link in links
        ]
        if not self._links:
            raise ValueError("FallbackChain requires at least one link")

    def run(
        self,
        chat_fn: Callable[..., GatewayResponse],
        /,
        **kwargs: Any,
    ) -> GatewayResponse:
        """Call `chat_fn` with each link until one returns. Raise GatewayError if all fail."""
        last_exc: Exception | None = None
        for link in self._links:
            try:
                return chat_fn(
                    model=link.primary,
                    fallback_models=link.fallbacks or None,
                    **kwargs,
                )
            except GatewayError as exc:
                log.warning(
                    "fallback: link %s+%s failed: %s",
                    link.primary,
                    list(link.fallbacks),
                    exc,
                )
                last_exc = exc
                continue
        raise GatewayError(
            f"All {len(self._links)} fallback links exhausted"
        ) from last_exc
