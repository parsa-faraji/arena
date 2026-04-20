"""Regression gate — compare a candidate run against a baseline.

Policy (deliberately simple; a CI gate should be easy to reason about):

  - Pull mean judge scores for both runs.
  - For every judge name present in the baseline, read the candidate's mean.
    A judge missing on the candidate is treated as a full regression — you
    promised this evaluator and didn't produce it.
  - A judge regresses if `candidate_mean < baseline_mean - threshold`.
    Threshold is an absolute delta in [0, 1], default 0.02.
  - If any judge regresses OR the candidate has more errored cases than the
    baseline, the gate fails.

The return value is a `GateReport` with the per-judge deltas and an overall
verdict. The CLI surface translates a failing verdict into exit code 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlmodel import select

from arena.store import CaseResult, JudgeScore, Run, session

if TYPE_CHECKING:
    from sqlalchemy import Engine


DEFAULT_THRESHOLD = 0.02


@dataclass(slots=True)
class JudgeDelta:
    judge: str
    baseline: float
    candidate: float | None
    delta: float  # candidate - baseline (None candidate → -baseline)
    regressed: bool


@dataclass(slots=True)
class GateReport:
    baseline_run_id: str
    candidate_run_id: str
    threshold: float
    passed: bool
    baseline_errors: int
    candidate_errors: int
    deltas: list[JudgeDelta] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def regressed_judges(self) -> list[JudgeDelta]:
        return [d for d in self.deltas if d.regressed]


def evaluate(
    *,
    engine: Engine,
    baseline_run_id: str,
    candidate_run_id: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> GateReport:
    """Compute the regression report. Does not raise on regression — the
    caller decides how loud to be."""
    baseline_means, baseline_errors = _run_summary(engine, baseline_run_id)
    candidate_means, candidate_errors = _run_summary(engine, candidate_run_id)

    deltas: list[JudgeDelta] = []
    for judge_name, baseline_score in sorted(baseline_means.items()):
        candidate_score = candidate_means.get(judge_name)
        if candidate_score is None:
            deltas.append(
                JudgeDelta(
                    judge=judge_name,
                    baseline=baseline_score,
                    candidate=None,
                    delta=-baseline_score,
                    regressed=True,
                )
            )
            continue
        delta = candidate_score - baseline_score
        deltas.append(
            JudgeDelta(
                judge=judge_name,
                baseline=baseline_score,
                candidate=candidate_score,
                delta=delta,
                regressed=delta < -threshold,
            )
        )

    notes: list[str] = []
    for judge_name in sorted(set(candidate_means) - set(baseline_means)):
        # New evaluators on the candidate are informational only — you can't
        # regress against a baseline that didn't score it.
        notes.append(f"new evaluator on candidate: {judge_name}")

    errors_regressed = candidate_errors > baseline_errors
    if errors_regressed:
        notes.append(
            f"candidate has more errored cases ({candidate_errors}) than "
            f"baseline ({baseline_errors})"
        )

    passed = not any(d.regressed for d in deltas) and not errors_regressed
    return GateReport(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        threshold=threshold,
        passed=passed,
        baseline_errors=baseline_errors,
        candidate_errors=candidate_errors,
        deltas=deltas,
        notes=notes,
    )


def _run_summary(engine: Engine, run_id: str) -> tuple[dict[str, float], int]:
    """Return (per-judge mean scores, errored case count) for a run."""
    with session(engine) as s:
        run = s.get(Run, run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        result_rows = s.exec(
            select(CaseResult.id, CaseResult.error).where(CaseResult.run_id == run_id)
        ).all()
        result_ids = [row[0] for row in result_rows]
        errors = sum(1 for row in result_rows if row[1])
        if not result_ids:
            return {}, errors
        score_rows = s.exec(
            select(JudgeScore.judge, JudgeScore.score).where(
                JudgeScore.result_id.in_(result_ids)  # type: ignore[attr-defined]
            )
        ).all()
    agg: dict[str, list[float]] = {}
    for judge_name, score in score_rows:
        agg.setdefault(judge_name, []).append(score)
    means = {name: sum(values) / len(values) for name, values in agg.items() if values}
    return means, errors


__all__ = [
    "DEFAULT_THRESHOLD",
    "GateReport",
    "JudgeDelta",
    "evaluate",
]
