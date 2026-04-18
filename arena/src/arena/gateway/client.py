"""Respan Gateway client (OpenAI-compatible).

Arena routes every LLM call through Respan's gateway so a Respan engineer
reviewing the project can see every inference in their own dashboard. The
gateway is OpenAI-compatible, so we reuse the `openai` SDK with an
overridden `base_url`. We add:

- a small, explicit `chat()` surface (no kwargs soup) that Arena's internals
  can depend on;
- a retry policy on transient errors;
- pass-through of Respan-specific `extra_body` fields (`fallback_models`,
  `disable_log`, custom variables);
- semantic cache hook points.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

Message = dict[str, Any]


class GatewayError(RuntimeError):
    """Raised when the gateway exhausts all retries and fallbacks."""


@dataclass(slots=True, frozen=True)
class GatewayResponse:
    """Lean, immutable view of a gateway reply.

    We deliberately don't expose the raw OpenAI response object — Arena's
    internal modules should only depend on this struct, which keeps the
    surface small enough that swapping providers is a one-file change.
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    raw: dict[str, Any]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GatewayClient:
    """Thin wrapper around the Respan OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.respan.ai/api",
        timeout_s: float = 60.0,
        default_model: str = "gpt-4o-mini",
        disable_log: bool = False,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,  # we retry ourselves via tenacity
        )
        self._default_model = default_model
        self._disable_log = disable_log

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        fallback_models: Iterable[str] | None = None,
        variables: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> GatewayResponse:
        """Send a chat completion request through Respan Gateway."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(
                (APIConnectionError, RateLimitError, httpx.TimeoutException)
            ),
        )
        def _send() -> GatewayResponse:
            return self._send_once(
                messages,
                model=model or self._default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                fallback_models=fallback_models,
                variables=variables,
                response_format=response_format,
            )

        try:
            return _send()
        except APIStatusError as exc:
            raise GatewayError(
                f"Respan Gateway returned {exc.status_code}: {exc.message}"
            ) from exc

    def _send_once(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        fallback_models: Iterable[str] | None,
        variables: dict[str, Any] | None,
        response_format: dict[str, Any] | None,
    ) -> GatewayResponse:
        extra_body: dict[str, Any] = {}
        if fallback_models:
            extra_body["fallback_models"] = list(fallback_models)
        if self._disable_log:
            extra_body["disable_log"] = True
        if variables:
            extra_body["variables"] = variables

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body:
            kwargs["extra_body"] = extra_body

        import time

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return GatewayResponse(
            content=content,
            model=resp.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
