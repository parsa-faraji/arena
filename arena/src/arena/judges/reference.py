"""Reference-based judge — does the output match a known-good answer?

Given a case with a `reference` field in `expected`, asks an LLM whether
the model's output says the same thing as the reference. Returns 0, 0.5
(partial), or 1 with rationale.

Useful when exact-match is too strict (wording varies) but you still
have a gold answer.
"""
from __future__ import annotations

from dataclasses import dataclass

from arena.evals.dataset import EvalCase
from arena.evals.evaluators import parse_json_output
from arena.gateway.client import GatewayClient
from arena.judges.base import JUDGE_SYSTEM_PROMPT, JudgeVerdict


_BUCKETS = {
    "yes": 1.0,
    "partial": 0.5,
    "no": 0.0,
}


@dataclass
class ReferenceJudge:
    """Compares model output to `case.expected[reference_field]`."""

    reference_field: str = "reference"
    name: str = "reference"
    model: str = "claude-haiku-4-5"
    temperature: float = 0.0

    def judge(
        self,
        case: EvalCase,
        output: str,
        *,
        client: GatewayClient,
    ) -> JudgeVerdict:
        reference = (case.expected or {}).get(self.reference_field)
        if not isinstance(reference, str) or not reference.strip():
            return JudgeVerdict(
                score=0.0,
                rationale="no reference supplied on case",
                raw={"skipped": True},
            )

        prompt = (
            "Decide whether the model output conveys the same key information as "
            "the reference. Ignore wording differences; focus on substance.\n\n"
            f"Reference:\n{reference}\n\n"
            f"Model output:\n{output}\n\n"
            'Return JSON: {"verdict": "yes"|"partial"|"no", "rationale": "short"}.'
        )
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
        verdict = str(parsed.get("verdict", "")).strip().lower()
        score = _BUCKETS.get(verdict, 0.0)
        return JudgeVerdict(
            score=score,
            rationale=str(parsed.get("rationale", "")),
            raw={"model": resp.model, **parsed},
        )
