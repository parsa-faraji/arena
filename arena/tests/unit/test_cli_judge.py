"""CLI wiring for `arena judge` — post-hoc and pairwise modes.

Both modes go through a real GatewayClient, so we monkey-patch the
factory to return a canned judge. The point of this test is to catch
wiring regressions (argument parsing, DB lookup, case hydration,
summary formatting), not to re-test the judges themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from arena.cli import app
from arena.gateway.client import GatewayResponse
from arena.store import Case, CaseResult, Run, Variant, create_engine, init_db, session


class _CannedJudgeClient:
    """GatewayClient stand-in that replies to judge calls with fixed JSON."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> GatewayResponse:
        self.calls += 1
        return GatewayResponse(
            content=self._reply,
            model="claude-haiku-4-5",
            input_tokens=10, output_tokens=5, latency_ms=1, raw={},
        )


def _seed_run(db_path: Path, run_name: str = "v0") -> tuple[str, str]:
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)
    with session(engine) as s:
        v = Variant(name=run_name, prompt="p", model="gpt-4o-mini")
        s.add(v); s.commit(); s.refresh(v)
        c = Case(id="case-1", dataset="demo", inputs_json='{"ticket": "refund please"}')
        s.add(c); s.commit()
        run = Run(variant_id=v.id, dataset="demo", status="done", total_cases=1, completed_cases=1)
        s.add(run); s.commit(); s.refresh(run)
        result = CaseResult(
            run_id=run.id, case_id="case-1",
            output='{"urgency": "high", "suggested_reply": "I\'ll refund"}',
            input_tokens=10, output_tokens=5, latency_ms=1, model="gpt-4o-mini",
        )
        s.add(result); s.commit()
    return run.id, v.id


def _write_config(path: Path) -> None:
    path.write_text(
        "evaluators:\n"
        "  - type: judge\n"
        "    judge: rubric\n"
        "    criterion: 'is the reply helpful?'\n"
        "    name: helpfulness\n"
    )


def test_judge_writes_scores_posthoc(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "arena.db"
    run_id, _ = _seed_run(db)
    _write_config(tmp_path / "arena.config.yaml")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    client = _CannedJudgeClient('{"score": 4, "rationale": "clear"}')
    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: client)
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(app, ["judge", run_id])
    assert result.exit_code == 0, result.stdout
    assert "wrote 1 judge scores" in result.stdout
    assert client.calls == 1


def test_judge_pairwise_reports_win_rate(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "arena.db"
    run_a, _ = _seed_run(db, "v0")
    # Seed a second run sharing the same case.
    engine = create_engine(f"sqlite:///{db}")
    with session(engine) as s:
        v = Variant(name="v1", prompt="p2", model="gpt-4o-mini")
        s.add(v); s.commit(); s.refresh(v)
        run_b = Run(variant_id=v.id, dataset="demo", status="done", total_cases=1, completed_cases=1)
        s.add(run_b); s.commit(); s.refresh(run_b)
        s.add(CaseResult(
            run_id=run_b.id, case_id="case-1", output='{"urgency": "high"}',
            input_tokens=10, output_tokens=5, latency_ms=1, model="gpt-4o-mini",
        ))
        s.commit()
        run_b_id = run_b.id

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    monkeypatch.setattr(
        arena_cli, "GatewayClient",
        lambda **_: _CannedJudgeClient('{"winner": "1", "rationale": "better"}'),
    )
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(app, ["judge", run_a, "--pairwise", "--vs", run_b_id])
    assert result.exit_code == 0, result.stdout
    assert "win rate" in result.stdout


def test_judge_requires_second_run_for_pairwise(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "arena.db"
    run_a, _ = _seed_run(db)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")

    from arena import cli as arena_cli

    monkeypatch.setattr(arena_cli, "GatewayClient", lambda **_: _CannedJudgeClient("{}"))
    monkeypatch.setattr(arena_cli, "init_tracing", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(app, ["judge", run_a, "--pairwise"])
    assert result.exit_code == 2
    assert "--vs" in result.stdout
