from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from arena.gateway import pricing
from arena.gateway.pricing import Price, cost_usd, is_known, register_price


def test_known_model_cost() -> None:
    # gpt-4o-mini: $0.15 in, $0.60 out per 1M tokens.
    # 1000 in, 500 out -> 0.15*0.001 + 0.60*0.0005 = 0.00015 + 0.00030 = 0.00045
    assert cost_usd("gpt-4o-mini", 1000, 500) == pytest.approx(0.00045, rel=1e-6)


def test_provider_prefix_stripped() -> None:
    assert cost_usd("openai/gpt-4o-mini", 1000, 500) == pytest.approx(0.00045, rel=1e-6)


def test_longest_prefix_match() -> None:
    # Unknown suffix -> falls back to the base model price.
    assert cost_usd("gpt-4o-mini-2025-05-01", 1000, 500) == pytest.approx(
        0.00045, rel=1e-6
    )


def test_unknown_model_returns_zero_and_logs_once(caplog) -> None:
    caplog.set_level("WARNING")
    pricing._unknown_models_logged.discard("unheard-of-model-v7")
    assert cost_usd("unheard-of-model-v7", 1000, 500) == 0.0
    assert not is_known("unheard-of-model-v7")
    # Calling again does not re-log.
    caplog.clear()
    cost_usd("unheard-of-model-v7", 1000, 500)
    assert all("unheard-of-model-v7" not in r.message for r in caplog.records)


def test_register_price_overrides_default() -> None:
    register_price("custom-model-x", Price(1.0, 2.0))
    assert cost_usd("custom-model-x", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "prices.json"
    override.write_text(
        json.dumps({"gpt-4o-mini": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})
    )
    monkeypatch.setenv("ARENA_PRICING", str(override))
    # Re-run the loader so the override takes effect in this test run.
    pricing._load_overrides()
    try:
        assert cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(3.0)
    finally:
        register_price("gpt-4o-mini", Price(0.15, 0.60))  # restore default
        os.environ.pop("ARENA_PRICING", None)
