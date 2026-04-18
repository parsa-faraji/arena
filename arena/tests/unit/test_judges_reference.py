from __future__ import annotations

from typing import Any

from arena.evals.dataset import EvalCase
from arena.gateway.client import GatewayResponse
from arena.judges.reference import ReferenceJudge


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> GatewayResponse:
        return GatewayResponse(
            content=self._reply, model="test", input_tokens=1, output_tokens=1,
            latency_ms=1, raw={},
        )


def test_reference_judge_yes() -> None:
    case = EvalCase(
        id="c", inputs={}, expected={"reference": "Refund the customer $50."}
    )
    client = _FakeClient('{"verdict": "yes", "rationale": "matches"}')
    v = ReferenceJudge().judge(case, "I'll refund the $50.", client=client)  # type: ignore[arg-type]
    assert v.score == 1.0


def test_reference_judge_partial() -> None:
    case = EvalCase(id="c", inputs={}, expected={"reference": "Refund $50."})
    client = _FakeClient('{"verdict": "partial", "rationale": "wrong amount"}')
    v = ReferenceJudge().judge(case, "I'll refund $20.", client=client)  # type: ignore[arg-type]
    assert v.score == 0.5


def test_reference_judge_skips_when_no_reference() -> None:
    case = EvalCase(id="c", inputs={}, expected={"category": "billing"})
    client = _FakeClient('{"verdict": "yes"}')
    v = ReferenceJudge().judge(case, "anything", client=client)  # type: ignore[arg-type]
    assert v.score == 0.0
    assert "no reference" in v.rationale


def test_reference_judge_unknown_verdict_is_zero() -> None:
    case = EvalCase(id="c", inputs={}, expected={"reference": "foo"})
    client = _FakeClient('{"verdict": "maybe?"}')
    v = ReferenceJudge().judge(case, "x", client=client)  # type: ignore[arg-type]
    assert v.score == 0.0
