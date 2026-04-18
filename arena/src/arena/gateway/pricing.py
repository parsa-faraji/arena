"""Per-model pricing in USD.

Used to surface cost in run summaries, the CI gate, and the dashboard.
Prices are list prices as of 2026-04 and are intended for budgeting and
comparison, not accounting — override via `ARENA_PRICING` (path to a
JSON file) or by calling `register_price()` at startup if you need exact
numbers (e.g. committed-use discounts).

Matching logic (in order):
  1. exact match on the model string after stripping provider prefix;
  2. exact match on the raw model string;
  3. longest-prefix match on the stripped string;
  4. return `None` (caller decides whether to fail loud or report $0).

A `None` return is deliberate — silently reporting $0 for a model we
don't know about would be misleading. Callers like the runner log a
warning the first time they hit an unpriced model.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Price:
    """USD per 1M tokens."""

    input_per_mtok: float
    output_per_mtok: float


# Defaults current as of 2026-04. Intentionally conservative rounding.
_DEFAULT_PRICES: dict[str, Price] = {
    # OpenAI
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4.1": Price(2.00, 8.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4.1-nano": Price(0.10, 0.40),
    "o3-mini": Price(1.10, 4.40),
    # Anthropic
    "claude-haiku-4-5": Price(0.80, 4.00),
    "claude-sonnet-4-5": Price(3.00, 15.00),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-opus-4-6": Price(15.00, 75.00),
    "claude-opus-4-7": Price(15.00, 75.00),
    # Google
    "gemini-2.5-flash": Price(0.30, 2.50),
    "gemini-2.5-pro": Price(1.25, 10.00),
}

_prices: dict[str, Price] = dict(_DEFAULT_PRICES)
_unknown_models_logged: set[str] = set()


def register_price(model: str, price: Price) -> None:
    """Register (or override) the price for a model. Case-insensitive."""
    _prices[model.lower()] = price


def _load_overrides() -> None:
    override_path = os.environ.get("ARENA_PRICING")
    if not override_path:
        return
    path = Path(override_path)
    if not path.exists():
        log.warning("ARENA_PRICING points to missing file: %s", path)
        return
    data = json.loads(path.read_text())
    for model, entry in data.items():
        _prices[model.lower()] = Price(
            input_per_mtok=float(entry["input_per_mtok"]),
            output_per_mtok=float(entry["output_per_mtok"]),
        )


_load_overrides()


def _lookup(model: str) -> Price | None:
    key = model.lower().strip()
    if key in _prices:
        return _prices[key]
    stripped = key.split("/")[-1] if "/" in key else key
    if stripped in _prices:
        return _prices[stripped]
    # longest-prefix match — e.g. "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini"
    candidates = [p for p in _prices if stripped.startswith(p)]
    if candidates:
        best = max(candidates, key=len)
        return _prices[best]
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost for a single call. Returns 0.0 for unknown models."""
    price = _lookup(model)
    if price is None:
        if model not in _unknown_models_logged:
            _unknown_models_logged.add(model)
            log.warning("no price registered for model %r; reporting $0", model)
        return 0.0
    return (
        (input_tokens / 1_000_000) * price.input_per_mtok
        + (output_tokens / 1_000_000) * price.output_per_mtok
    )


def is_known(model: str) -> bool:
    return _lookup(model) is not None
