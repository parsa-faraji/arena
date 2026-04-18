"""CLI smoke tests for `arena runs` and `arena show`.

We seed a run directly into the DB and verify the output contains the
expected run id prefix + variant name. This lets us exercise the read
path without spinning up a gateway client.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from arena.cli import app
from arena.store import CaseResult, JudgeScore, Run, Variant, create_engine, init_db, session

runner = CliRunner()


def _seed(db_path: Path) -> tuple[str, str]:
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)
    with session(engine) as s:
        v = Variant(name="v0", prompt="triage", model="gpt-4o-mini")
        s.add(v)
        s.commit()
        s.refresh(v)
        run = Run(
            variant_id=v.id,
            dataset="demo",
            status="done",
            total_cases=2,
            completed_cases=2,
        )
        s.add(run)
        s.commit()
        s.refresh(run)

        result = CaseResult(
            run_id=run.id,
            case_id="case-1",
            output='{"urgency": "low"}',
            input_tokens=10,
            output_tokens=5,
            latency_ms=123,
            model="gpt-4o-mini",
            cache_hit=False,
        )
        s.add(result)
        s.commit()
        s.refresh(result)
        s.add(JudgeScore(result_id=result.id, judge="exact_match:urgency", score=1.0))
        s.commit()
    return run.id, v.id


def test_runs_lists_recent(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "runs.db"
    run_id, _ = _seed(db)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")
    result = runner.invoke(app, ["runs", "--limit", "5"])
    assert result.exit_code == 0, result.stdout
    assert run_id[:8] in result.stdout
    assert "v0" in result.stdout
    assert "done" in result.stdout


def test_show_by_prefix(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "runs.db"
    run_id, _ = _seed(db)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")
    result = runner.invoke(app, ["show", run_id[:8]])
    assert result.exit_code == 0, result.stdout
    assert run_id in result.stdout
    assert "v0" in result.stdout
    assert "exact_match:urgency" in result.stdout


def test_show_unknown_run_errors(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "runs.db"
    _seed(db)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")
    result = runner.invoke(app, ["show", "nonexistent"])
    assert result.exit_code == 1
    assert "no run matching" in result.stdout
