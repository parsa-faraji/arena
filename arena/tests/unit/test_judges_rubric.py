from __future__ import annotations

from typing import Any

import pytest

from arena.evals.dataset import EvalCase
from arena.gateway.client import GatewayResponse
from arena.judges.base import JudgeVerdict
from arena.judges.rubric import RubricJudge, _to_unit


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_prompt: str | None = None
        self.last_model: str | None = None
        self.last_response_format: dict[str, Any] | None = None

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> GatewayResponse:
        self.last_prompt = messages[-1]["content"]
        self.last_model = kwargs.get("model")
        self.last_response_format = kwargs.get("response_format")
        return GatewayResponse(
            content=self._reply,
            model=kwargs.get("model", "test-model"),
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            raw={},
        )


def test_to_unit_converts_likert() -> None:
    assert _to_unit(1) == 0.0
    assert _to_unit(3) == 0.5
    assert _to_unit(5) == 1.0
    assert _to_unit(0.8) == 0.8
    assert _to_unit("garbage") == 0.0


def test_rubric_judge_parses_score_and_rationale() -> None:
    client = _FakeClient('{"score": 4, "rationale": "clear but a bit terse"}')
    judge = RubricJudge(criterion="is the reply helpful?", name="helpfulness")
    v = judge.judge(
        EvalCase(id="c1", inputs={"ticket": "refund please"}),
        output="I'll refund you",
        client=client,  # type: ignore[arg-type]
    )
    assert isinstance(v, JudgeVerdict)
    assert v.score == 0.75
    assert "clear" in v.rationale
    assert client.last_response_format == {"type": "json_object"}
    assert "refund please" in (client.last_prompt or "")


def test_rubric_judge_scopes_to_field_when_configured() -> None:
    client = _FakeClient('{"score": 5, "rationale": "good"}')
    judge = RubricJudge(
        criterion="rate tone",
        name="tone",
        target_field="suggested_reply",
    )
    model_output = '{"urgency": "high", "suggested_reply": "I can help with that."}'
    judge.judge(
        EvalCase(id="c", inputs={"ticket": "issue"}),
        output=model_output,
        client=client,  # type: ignore[arg-type]
    )
    assert "I can help with that." in (client.last_prompt or "")
    assert '"urgency"' not in (client.last_prompt or "")


def test_rubric_judge_falls_back_to_full_output_when_field_missing() -> None:
    client = _FakeClient('{"score": 2, "rationale": "meh"}')
    judge = RubricJudge(criterion="x", name="x", target_field="missing_field")
    model_output = '{"urgency": "high"}'
    judge.judge(
        EvalCase(id="c", inputs={"ticket": "t"}),
        output=model_output,
        client=client,  # type: ignore[arg-type]
    )
    assert model_output in (client.last_prompt or "")


def test_rubric_judge_score_in_range() -> None:
    client = _FakeClient('{"score": "not a number", "rationale": ""}')
    judge = RubricJudge(criterion="x", name="x")
    v = judge.judge(
        EvalCase(id="c", inputs={}),
        output="anything",
        client=client,  # type: ignore[arg-type]
    )
    assert v.score == 0.0


def test_judge_verdict_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"must be in"):
        JudgeVerdict(score=1.5, rationale="x")
