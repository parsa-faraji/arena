"""CLI wiring for `arena optimize`.

We stub the gateway so the test is offline and deterministic. The fake
responds to triage calls with a terse v0-ish reply by default, a richer
v1 reply once the prompt contains the marker the optimizer proposes,
and scores outputs by length (long = 5, short = 2) so the optimizer
actually promotes the proposal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlmodel import select
from typer.testing import CliRunner

from arena.cli import app
from arena.gateway.client import GatewayResponse
from arena.store import OptimizerStep, create_engine, session

_DATASET = [
    {"id": f"case-{i:02d}", "inputs": {"ticket": "refund please"}}
    for i in range(1, 6)
]

_V1_MARKER = "Decisive reply with next step"


class _OptimizeGateway:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> GatewayResponse:
        self.calls += 1
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user = messages[-1]["content"] if messages else ""

        if "prompt engineer" in system.lower():
            content = json.dumps(
                {
                    "prompt": (
                        f"{_V1_MARKER}. You are a support triage assistant. "
                        "Return a JSON with urgency/category/suggested_reply."
                    ),
                    "gradient": "add actionable next-step language",
                }
            )
        elif "strict but fair evaluator" in system.lower():
            # Longer embedded output => higher score (proxy for "more helpful").
            score = 5 if "within a few hours" in user else 2
            content = json.dumps({"score": score, "rationale": "ok"})
        else:
            if _V1_MARKER in system:
                reply = (
                    "Thanks for flagging — I've queued your refund, "
                    "you'll see confirmation in your inbox within a few hours."
                )
            else:
                reply = "Got it — I'll take a look."
            content = json.dumps(
                {"urgency": "high", "category": "billing", "suggested_reply": reply}
            )

        return GatewayResponse(
            content=content,
            model="gpt-4o-mini",
            input_tokens=40,
            output_tokens=20,
            latency_ms=1,
            raw={},
        )


def _write_support_project(project_dir: Path) -> None:
    (project_dir / "prompts").mkdir()
    (project_dir / "prompts" / "v0.md").write_text(
        "You are a support triage assistant. Return JSON.\n"
    )
    (project_dir / "dataset.jsonl").write_text(
        "\n".join(json.dumps(c) for c in _DATASET) + "\n"
    )
    (project_dir / "arena.config.yaml").write_text(
        "default_model: gpt-4o-mini\n"
        "optimizer_model: gpt-4o-mini\n"
        "evaluators:\n"
        "  - type: judge\n"
        "    judge: rubric\n"
        "    name: reply_quality\n"
        "    criterion: is the reply helpful?\n"
    )


def test_optimize_promotes_better_prompt(tmp_path: Path, monkeypatch) -> None:
    _write_support_project(tmp_path)
    db = tmp_path / "arena.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: _OptimizeGateway())
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "optimize",
            "v0",
            "--budget",
            "2",
            "--cases",
            "5",
            "--target",
            "reply_quality",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "promoted" in result.stdout
    assert "yes" in result.stdout

    # The winning prompt should land on disk for re-runs.
    assert (tmp_path / "prompts" / "v0-optimized.md").exists()

    # At least one OptimizerStep row should be marked kept.
    engine = create_engine(f"sqlite:///{db}")
    with session(engine) as s:
        steps = s.exec(select(OptimizerStep)).all()
        assert steps, "optimizer did not persist any steps"
        assert any(step.kept for step in steps), "no proposal was promoted"


def test_optimize_errors_when_target_missing(tmp_path: Path, monkeypatch) -> None:
    _write_support_project(tmp_path)
    db = tmp_path / "arena.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: _OptimizeGateway())
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["optimize", "v0", "--budget", "1", "--cases", "2", "--target", "ghost"],
    )
    assert result.exit_code == 2
    assert "not found" in result.stdout
