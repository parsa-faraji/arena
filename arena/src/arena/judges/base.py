"""Judge protocol — the shared surface every LLM-as-judge implementation exposes.

Design notes:
- A `Judge` is given a `(case, output, client)` triple and returns a
  `JudgeVerdict` with a normalised 0..1 score, a short rationale, and the
  raw JSON the LLM produced.
- Judges take the gateway client as an argument rather than holding one
  on `self`, so a single judge instance can be reused across runs and
  the test suite can inject stubs. The runner's `JudgeEvaluator` adapter
  captures the client once and satisfies the `Evaluator` interface from
  the outside.
- The `model` and `temperature` live on the judge, not the client —
  different judges in an ensemble typically want different models.
- Every judge call routes through Respan Gateway, so a reviewer can
  literally scroll through the judge reasoning in the Respan trace UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from arena.evals.dataset import EvalCase
from arena.gateway.client import GatewayClient


@dataclass(slots=True)
class JudgeVerdict:
    """One judge's opinion about one (case, output) pair."""

    score: float  # in [0, 1]
    rationale: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


class Judge(Protocol):
    """Scores a single (case, output) pair using an LLM."""

    name: str

    def judge(
        self,
        case: EvalCase,
        output: str,
        *,
        client: GatewayClient,
    ) -> JudgeVerdict: ...


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator. Respond ONLY with a JSON object "
    "matching the schema described in the user message. Do not include prose "
    "before or after the JSON."
)
