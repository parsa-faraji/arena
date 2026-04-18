"""End-to-end smoke of `arena run` against a fake gateway.

This exercises the full CLI path: loading arena.config.yaml, building
evaluators, spinning up the runner, writing to SQLite, and printing the
summary table. If the CLI wiring drifts, this test catches it without
needing a real Respan account.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arena.cli import app
from arena.gateway.client import GatewayResponse


class _FakeGateway:
    """Drop-in for GatewayClient with a canned reply."""

    def __init__(self, **_: Any) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> GatewayResponse:
        self.calls += 1
        return GatewayResponse(
            content='{"urgency": "high", "category": "billing", "suggested_reply": "refund"}',
            model="gpt-4o-mini",
            input_tokens=30,
            output_tokens=20,
            latency_ms=12,
            raw={},
        )


def _write_project(root: Path) -> None:
    (root / "prompts").mkdir()
    (root / "prompts" / "v0.md").write_text("triage this.")
    (root / "dataset.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"c{i}",
                    "inputs": {"ticket": f"case {i}"},
                    "expected": {"urgency": "high", "category": "billing"},
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    (root / "arena.config.yaml").write_text(
        "evaluators:\n"
        "  - type: exact_match\n"
        "    field: urgency\n"
        "  - type: json_parse\n"
        "    fields: [urgency, category, suggested_reply]\n"
    )


def test_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_project(tmp_path)
    db_path = tmp_path / "arena.db"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Swap the gateway before the CLI constructs one.
    from arena import cli as arena_cli

    fake = _FakeGateway()
    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: fake)

    # Skip tracing init — it tries to load the real SDK.
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "v0", "--cases", "3", "--concurrency", "2"])
    assert result.exit_code == 0, result.stdout

    assert "exact_match:urgency" in result.stdout
    assert "json_parse" in result.stdout
    assert "Run id:" in result.stdout
    assert "cost" in result.stdout
    # 3 cases distinct -> 3 gateway calls (no cache hits expected)
    assert fake.calls == 3

    # Data should be persisted too — list the run via `arena runs`.
    result = runner.invoke(app, ["runs", "--limit", "5"])
    assert result.exit_code == 0, result.stdout
    assert "v0" in result.stdout
    assert "done" in result.stdout


def test_run_with_missing_prompt_errors(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: _FakeGateway())
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "missing-variant"])
    assert result.exit_code == 1
    assert "Missing prompt file" in result.stdout


def test_run_with_invalid_config_fails_loud(tmp_path: Path, monkeypatch) -> None:
    _write_project(tmp_path)
    # Overwrite with a typo'd evaluator type.
    (tmp_path / "arena.config.yaml").write_text(
        "evaluators:\n  - type: exact_matchx\n    field: urgency\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: _FakeGateway())
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "v0", "--cases", "1"])
    assert result.exit_code == 1
    assert "invalid config" in result.stdout or "exact_matchx" in result.stdout
