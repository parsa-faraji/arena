"""Rubric-based judge — G-Eval style scoring.

Prompts an LLM to rate a model output against a named rubric, returning a
1..5 score plus rationale. Scores are normalised to [0, 1] downstream.

Why 1..5 rather than 0..1 out of the gate? LLMs (especially smaller
models) are notably better at picking a discrete Likert-style number than
calibrating a continuous score. We convert to [0, 1] by `(score - 1) / 4`.
"""
from __future__ import annotations

from dataclasses import dataclass

from arena.evals.dataset import EvalCase
from arena.evals.evaluators import parse_json_output
from arena.gateway.client import GatewayClient
from arena.judges.base import JUDGE_SYSTEM_PROMPT, JudgeVerdict


@dataclass
class RubricJudge:
    """Single-criterion Likert judge.

    Example::

        judge = RubricJudge(
            name="helpfulness",
            criterion="Does the suggested reply solve the customer's issue?",
            model="claude-haiku-4-5",
        )
    """

    criterion: str
    name: str = ""
    model: str = "claude-haiku-4-5"
    temperature: float = 0.0
    # If set, judge only the named field of a JSON model output.
    target_field: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "rubric"
        else:
            self.name = f"rubric:{self.name}"

    def judge(
        self,
        case: EvalCase,
        output: str,
        *,
        client: GatewayClient,
    ) -> JudgeVerdict:
        target = self._extract_target(output)
        prompt = self._build_prompt(case, target)
        resp = client.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_output(resp.content) or {}
        raw_score = parsed.get("score")
        rationale = str(parsed.get("rationale", "")).strip()

        normalised = _to_unit(raw_score)
        return JudgeVerdict(
            score=normalised,
            rationale=rationale,
            raw={"model": resp.model, **parsed},
        )

    # ------------------------------------------------------------ helpers

    def _extract_target(self, output: str) -> str:
        if self.target_field is None:
            return output
        parsed = parse_json_output(output) or {}
        value = parsed.get(self.target_field)
        if isinstance(value, str):
            return value
        return output  # fall back to whole reply if the field isn't present

    def _build_prompt(self, case: EvalCase, target: str) -> str:
        return (
            f"You are judging a model's output against this rubric:\n"
            f"{self.criterion}\n\n"
            f"Input given to the model:\n{case.user_text()}\n\n"
            f"Model output to judge:\n{target}\n\n"
            'Return JSON: {"score": integer 1..5, "rationale": "one short sentence"}.\n'
            "1 = fails the rubric completely. 5 = clearly meets it. Use the full range."
        )


def _to_unit(raw: object) -> float:
    """Convert a Likert 1..5 (or already-normalised 0..1) score to [0, 1]."""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    # Integer values 1..5 are unambiguously Likert — don't mistake a Likert 1
    # for "100% score". We check the raw type because float(1) == 1.0 loses it.
    if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= 5:
        return (value - 1.0) / 4.0
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 5.0:
        return (value - 1.0) / 4.0
    return max(0.0, min(1.0, value))
