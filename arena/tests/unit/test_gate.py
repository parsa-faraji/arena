"""Regression gate unit tests.

Exercises both the pure ``evaluate`` function and the ``arena gate`` CLI
wrapper. Uses an in-memory DB seeded with two runs sharing the same
judge names but different mean scores.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from arena.cli import app
from arena.gate import evaluate
from arena.store import (
    CaseResult,
    JudgeScore,
    Run,
    Variant,
    create_engine,
    init_db,
    session,
)


def _seed_run(
    engine,
    *,
    variant_name: str,
    scores_by_case: dict[str, dict[str, float]],
    errored_cases: list[str] | None = None,
) -> str:
    """Seed one run with ``scores_by_case[case_id][judge_name] = score``.
    Returns the run id."""
    errored = set(errored_cases or [])
    with session(engine) as s:
        variant = Variant(name=variant_name, prompt="", model="gpt-4o-mini")
        s.add(variant)
        s.commit()
        s.refresh(variant)
        run = Run(
            variant_id=variant.id,
            dataset="demo",
            status="done",
            total_cases=len(scores_by_case),
            completed_cases=len(scores_by_case),
        )
        s.add(run)
        s.commit()
        s.refresh(run)

        for case_id, judge_scores in scores_by_case.items():
            result = CaseResult(
                run_id=run.id,
                case_id=case_id,
                output="{}",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                model="gpt-4o-mini",
                error="boom" if case_id in errored else None,
            )
            s.add(result)
            s.flush()
            for judge_name, score in judge_scores.items():
                s.add(
                    JudgeScore(result_id=result.id, judge=judge_name, score=score)
                )
        s.commit()
        return run.id


def _make_engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'arena.db'}")
    init_db(engine)
    return engine


def test_gate_passes_when_candidate_beats_baseline(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    baseline = _seed_run(
        engine,
        variant_name="v0",
        scores_by_case={
            "case-1": {"helpfulness": 0.4, "json_parse": 1.0},
            "case-2": {"helpfulness": 0.5, "json_parse": 1.0},
        },
    )
    candidate = _seed_run(
        engine,
        variant_name="v1",
        scores_by_case={
            "case-1": {"helpfulness": 0.9, "json_parse": 1.0},
            "case-2": {"helpfulness": 0.9, "json_parse": 1.0},
        },
    )
    report = evaluate(
        engine=engine, baseline_run_id=baseline, candidate_run_id=candidate
    )
    assert report.passed
    assert report.regressed_judges == []


def test_gate_fails_on_regression(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    baseline = _seed_run(
        engine,
        variant_name="v1",
        scores_by_case={
            "case-1": {"helpfulness": 0.9},
            "case-2": {"helpfulness": 0.9},
        },
    )
    candidate = _seed_run(
        engine,
        variant_name="v0",
        scores_by_case={
            "case-1": {"helpfulness": 0.3},
            "case-2": {"helpfulness": 0.4},
        },
    )
    report = evaluate(
        engine=engine, baseline_run_id=baseline, candidate_run_id=candidate
    )
    assert not report.passed
    assert len(report.regressed_judges) == 1
    assert report.regressed_judges[0].judge == "helpfulness"


def test_gate_flags_errored_cases(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    baseline = _seed_run(
        engine,
        variant_name="v1",
        scores_by_case={"case-1": {"h": 0.8}, "case-2": {"h": 0.8}},
    )
    candidate = _seed_run(
        engine,
        variant_name="v2",
        scores_by_case={"case-1": {"h": 0.8}, "case-2": {"h": 0.8}},
        errored_cases=["case-1"],
    )
    report = evaluate(
        engine=engine, baseline_run_id=baseline, candidate_run_id=candidate
    )
    assert not report.passed
    assert any("errored" in note for note in report.notes)


def test_gate_reports_missing_judge_as_regression(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    baseline = _seed_run(
        engine,
        variant_name="v1",
        scores_by_case={"case-1": {"helpfulness": 0.9, "clarity": 0.9}},
    )
    candidate = _seed_run(
        engine,
        variant_name="v2",
        scores_by_case={"case-1": {"helpfulness": 0.9}},  # clarity missing
    )
    report = evaluate(
        engine=engine, baseline_run_id=baseline, candidate_run_id=candidate
    )
    assert not report.passed
    missing = [d for d in report.deltas if d.candidate is None]
    assert len(missing) == 1
    assert missing[0].judge == "clarity"


def test_gate_cli_exit_code_matches_verdict(tmp_path: Path, monkeypatch) -> None:
    engine = _make_engine(tmp_path)
    baseline = _seed_run(
        engine,
        variant_name="v1",
        scores_by_case={"case-1": {"h": 0.9}},
    )
    candidate_good = _seed_run(
        engine,
        variant_name="v2",
        scores_by_case={"case-1": {"h": 0.95}},
    )
    candidate_bad = _seed_run(
        engine,
        variant_name="v3",
        scores_by_case={"case-1": {"h": 0.3}},
    )

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'arena.db'}")
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test")
    runner = CliRunner()

    ok = runner.invoke(app, ["gate", "--baseline", baseline, "--run", candidate_good])
    assert ok.exit_code == 0, ok.stdout
    assert "PASS" in ok.stdout

    fail = runner.invoke(app, ["gate", "--baseline", baseline, "--run", candidate_bad])
    assert fail.exit_code == 1, fail.stdout
    assert "FAIL" in fail.stdout
