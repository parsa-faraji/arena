from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from arena.evals.dataset import Dataset, EvalCase
from arena.evals.evaluators import ExactMatchEvaluator, JSONParseEvaluator
from arena.evals.runner import RunConfig, VariantRunner
from arena.gateway.cache import SemanticCache
from arena.gateway.client import GatewayResponse
from arena.store import Variant, create_engine, init_db


class FakeClient:
    """Stand-in for GatewayClient that returns a canned reply per case."""

    def __init__(self, reply_for: dict[str, str]) -> None:
        self._reply_for = reply_for
        self.calls = 0
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **_: Any,
    ) -> GatewayResponse:
        with self._lock:
            self.calls += 1
        user = next(m["content"] for m in messages if m["role"] == "user")
        content = self._reply_for.get(user, '{"urgency": "low", "category": "other"}')
        time.sleep(0.005)  # simulate latency so concurrency is testable
        return GatewayResponse(
            content=content,
            model=model,
            input_tokens=10,
            output_tokens=5,
            latency_ms=5,
            raw={},
        )


def _make_engine(tmp_path: Path) -> Any:
    engine = create_engine(f"sqlite:///{tmp_path / 'runner.db'}")
    init_db(engine)
    return engine


def test_runner_scores_and_persists(tmp_path: Path) -> None:
    cases = [
        EvalCase(id="c1", inputs={"ticket": "charge twice"}, expected={"urgency": "high"}),
        EvalCase(id="c2", inputs={"ticket": "dark mode?"}, expected={"urgency": "low"}),
    ]
    ds = Dataset.from_cases(cases, name="demo")
    client = FakeClient(
        {
            "charge twice": '{"urgency": "high", "category": "billing"}',
            "dark mode?": '{"urgency": "low", "category": "feature"}',
        }
    )
    engine = _make_engine(tmp_path)
    runner = VariantRunner(client=client, engine=engine)  # type: ignore[arg-type]

    variant = Variant(name="v0", prompt="triage this", model="gpt-4o-mini")
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
    assert summary.completed_cases == 2
    assert summary.errors == 0
    assert summary.per_evaluator["exact_match:urgency"] == 1.0
    assert summary.per_evaluator["json_parse"] == 1.0
    assert summary.total_input_tokens == 20
    assert client.calls == 2


def test_runner_uses_cache(tmp_path: Path) -> None:
    cases = [
        EvalCase(id=f"c{i}", inputs={"ticket": "same ticket"}, expected={"urgency": "low"})
        for i in range(5)
    ]
    ds = Dataset.from_cases(cases, name="cached")
    client = FakeClient({"same ticket": '{"urgency": "low"}'})
    cache = SemanticCache(enable_semantic=False)
    engine = _make_engine(tmp_path)
    runner = VariantRunner(client=client, engine=engine)  # type: ignore[arg-type]

    variant = Variant(name="v0", prompt="triage", model="gpt-4o-mini")
    summary = runner.run(
        ds,
        RunConfig(
            variant=variant,
            evaluators=[ExactMatchEvaluator(field="urgency")],
            max_concurrency=1,
            cache=cache,
        ),
    )
    assert summary.completed_cases == 5
    # 5 cases but only 1 real gateway call — the rest hit the exact cache.
    assert client.calls == 1
    assert summary.cache_hits == 4


def test_runner_survives_individual_case_errors(tmp_path: Path) -> None:
    cases = [
        EvalCase(id="ok", inputs={"ticket": "ok"}, expected={"urgency": "low"}),
        EvalCase(id="bad", inputs={"ticket": "bad"}, expected={"urgency": "low"}),
    ]
    ds = Dataset.from_cases(cases, name="mixed")

    class FlakyClient(FakeClient):
        def chat(self, messages, **kwargs):  # type: ignore[override]
            user = next(m["content"] for m in messages if m["role"] == "user")
            if user == "bad":
                from arena.gateway.client import GatewayError

                raise GatewayError("simulated 500")
            return super().chat(messages, **kwargs)

    client = FlakyClient({"ok": '{"urgency": "low"}'})
    engine = _make_engine(tmp_path)
    runner = VariantRunner(client=client, engine=engine)  # type: ignore[arg-type]

    variant = Variant(name="v0", prompt="triage", model="gpt-4o-mini")
    summary = runner.run(
        ds,
        RunConfig(
            variant=variant,
            evaluators=[ExactMatchEvaluator(field="urgency")],
            max_concurrency=2,
        ),
    )
    assert summary.errors == 1
    assert summary.completed_cases == 2
