"""Evaluators — score a model output against an EvalCase.

We deliberately keep the interface tiny: each evaluator is a callable that
takes `(case, output)` and returns an `EvaluatorResult`. That lets us mix
handwritten evaluators, regex checks, and LLM-as-judge evaluators through
the same pipeline. Judge evaluators plug in here through the
`JudgeEvaluator` wrapper below, which delegates to `arena.judges`.

Every evaluator has a `name` so the DB / dashboard can attribute scores.
Scores are floats in [0, 1]; `passed` is a convenience boolean derived
from a threshold (default 1.0 for exact-match, 0.5 for graded).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from arena.evals.dataset import EvalCase

if TYPE_CHECKING:
    from arena.gateway.client import GatewayClient
    from arena.judges.base import Judge


@dataclass(slots=True)
class EvaluatorResult:
    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    name: str

    def score(self, case: EvalCase, output: str) -> EvaluatorResult: ...


# ---------------------------------------------------------------- helpers


def parse_json_output(output: str) -> dict[str, Any] | None:
    """Robustly pull a JSON object out of a model's reply.

    Tries:
      1. direct json.loads
      2. strip triple-backtick code fences, retry
      3. regex the first top-level `{...}` block, retry

    Returns None if nothing parses — the evaluator then treats the case as
    a zero score with `details={"parse_error": ...}` so we can debug it.
    """
    text = output.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------- evaluators


@dataclass(slots=True)
class ExactMatchEvaluator:
    """Case-insensitive exact match on a single field in the expected dict.

    If the output parses as JSON we compare `output_dict[field]` to
    `case.expected[field]`. Otherwise we compare the raw trimmed output.
    """

    field: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"exact_match:{self.field}"

    def score(self, case: EvalCase, output: str) -> EvaluatorResult:
        expected = (case.expected or {}).get(self.field)
        if expected is None:
            return EvaluatorResult(
                self.name, 0.0, False, {"skipped": "no expected value"}
            )
        parsed = parse_json_output(output)
        actual: Any
        if parsed is not None and self.field in parsed:
            actual = parsed[self.field]
        else:
            actual = output.strip()

        passed = str(actual).strip().lower() == str(expected).strip().lower()
        return EvaluatorResult(
            self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"expected": expected, "actual": actual},
        )


@dataclass(slots=True)
class JSONParseEvaluator:
    """Pass iff the output parses to a JSON object containing all required fields."""

    required_fields: tuple[str, ...]
    name: str = "json_parse"

    def score(self, case: EvalCase, output: str) -> EvaluatorResult:
        parsed = parse_json_output(output)
        if parsed is None:
            return EvaluatorResult(self.name, 0.0, False, {"parse_error": True})
        missing = [f for f in self.required_fields if f not in parsed]
        passed = not missing
        return EvaluatorResult(
            self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"missing": missing},
        )


@dataclass
class RegexEvaluator:
    """Pass iff the output matches the pattern at least once."""

    pattern: str
    name: str = ""
    flags: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"regex:{self.pattern}"
        self._compiled = re.compile(self.pattern, self.flags)

    def score(self, case: EvalCase, output: str) -> EvaluatorResult:
        match = self._compiled.search(output)
        passed = match is not None
        return EvaluatorResult(
            self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"matched": match.group(0) if match else None},
        )


class JudgeEvaluator:
    """Adapts an LLM-as-judge to the `Evaluator` surface.

    The judge does the LLM call; this class wraps its verdict in the
    `EvaluatorResult` shape the runner persists. Kept class-based rather
    than a dataclass so the name can be synthesised at construction time.
    """

    def __init__(
        self,
        judge: Judge,
        client: GatewayClient,
        *,
        name: str | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        self._judge = judge
        self._client = client
        self.name = name or judge.name
        self._pass_threshold = pass_threshold

    def score(self, case: EvalCase, output: str) -> EvaluatorResult:
        verdict = self._judge.judge(case, output, client=self._client)
        details: dict[str, Any] = {"rationale": verdict.rationale}
        details.update(verdict.raw)
        return EvaluatorResult(
            name=self.name,
            score=verdict.score,
            passed=verdict.score >= self._pass_threshold,
            details=details,
        )
