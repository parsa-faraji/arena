"""The runner must persist CaseResults AND per-evaluator scores.

Without this, a completed run loses its numeric summary the moment the
process exits — the dashboard and the CI gate would have nothing to diff
against.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from sqlmodel import select

from arena.evals.dataset import Dataset, EvalCase
from arena.evals.evaluators import ExactMatchEvaluator, JSONParseEvaluator
from arena.evals.runner import RunConfig, VariantRunner
from arena.gateway.client import GatewayResponse
from arena.store import CaseResult, JudgeScore, Variant, create_engine, init_db, session


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> GatewayResponse:
        with self._lock:
            self.calls += 1
        return GatewayResponse(
            content='{"urgency": "low", "category": "other"}',
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=5,
            latency_ms=5,
            raw={},
        )


def test_persists_case_results_and_scores(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    init_db(engine)

    ds = Dataset.from_cases(
        [
            EvalCase(id="c1", inputs={"ticket": "hi"}, expected={"urgency": "low"}),
            EvalCase(id="c2", inputs={"ticket": "ok"}, expected={"urgency": "low"}),
        ],
        name="demo",
    )
    variant = Variant(name="v0", prompt="triage", model="gpt-4o-mini")
    runner = VariantRunner(client=FakeClient(), engine=engine)  # type: ignore[arg-type]

    summary = runner.run(
        ds,
        RunConfig(
            variant=variant,
            evaluators=[
                ExactMatchEvaluator(field="urgency"),
                JSONParseEvaluator(required_fields=("urgency", "category")),
            ],
            max_concurrency=2,
        ),
    )

    with session(engine) as s:
        results = s.exec(select(CaseResult).where(CaseResult.run_id == summary.run_id)).all()
        assert len(results) == 2
        # Each result gets one JudgeScore row per evaluator.
        result_ids = [r.id for r in results]
        scores = s.exec(
            select(JudgeScore).where(JudgeScore.result_id.in_(result_ids))  # type: ignore[attr-defined]
        ).all()
        assert len(scores) == 4
        judge_names = {score.judge for score in scores}
        assert judge_names == {"exact_match:urgency", "json_parse"}


def test_variant_is_persisted_once(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    init_db(engine)

    variant = Variant(name="v0", prompt="p", model="gpt-4o-mini")
    runner = VariantRunner(client=FakeClient(), engine=engine)  # type: ignore[arg-type]
    ds = Dataset.from_cases(
        [EvalCase(id="a", inputs={"ticket": "a"}, expected={"urgency": "low"})],
        name="demo",
    )

    runner.run(ds, RunConfig(variant=variant, evaluators=[], max_concurrency=1))
    runner.run(ds, RunConfig(variant=variant, evaluators=[], max_concurrency=1))

    with session(engine) as s:
        rows = s.exec(select(Variant).where(Variant.id == variant.id)).all()
        assert len(rows) == 1  # second run reused the existing variant row
