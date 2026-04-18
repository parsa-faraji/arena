"""Pairwise judge — "is A better than B?" for the same input.

The canonical use: after running two prompt variants on the same dataset,
pick which one wins per case, then aggregate into a win rate. This is
what the optimizer consumes when deciding whether to promote a proposal.

Position bias is a known issue — models systematically prefer the first
option. We mitigate by randomising the presentation order and mapping
the verdict back to the original A/B.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from arena.evals.dataset import EvalCase
from arena.evals.evaluators import parse_json_output
from arena.gateway.client import GatewayClient
from arena.judges.base import JUDGE_SYSTEM_PROMPT


Verdict = Literal["a", "b", "tie"]


@dataclass(slots=True)
class PairwiseResult:
    case_id: str
    verdict: Verdict
    rationale: str
    presented_first: Verdict  # which of A/B was shown first to the judge


@dataclass
class PairwiseJudge:
    """Compares two outputs for the same case and returns a winner."""

    criterion: str
    name: str = "pairwise"
    model: str = "claude-haiku-4-5"
    temperature: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def compare(
        self,
        case: EvalCase,
        output_a: str,
        output_b: str,
        *,
        client: GatewayClient,
    ) -> PairwiseResult:
        swap = self.rng.random() < 0.5
        first, second = (output_b, output_a) if swap else (output_a, output_b)
        first_label, second_label = ("b", "a") if swap else ("a", "b")

        prompt = (
            f"You are picking the better of two model outputs for this input:\n"
            f"{case.user_text()}\n\n"
            f"Criterion: {self.criterion}\n\n"
            f"Output 1:\n{first}\n\n"
            f"Output 2:\n{second}\n\n"
            'Return JSON: {"winner": "1"|"2"|"tie", "rationale": "one sentence"}.'
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
        raw_winner = str(parsed.get("winner", "")).strip()
        verdict: Verdict
        if raw_winner == "1":
            verdict = first_label  # type: ignore[assignment]
        elif raw_winner == "2":
            verdict = second_label  # type: ignore[assignment]
        else:
            verdict = "tie"

        return PairwiseResult(
            case_id=case.id,
            verdict=verdict,
            rationale=str(parsed.get("rationale", "")),
            presented_first=first_label,  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class PairwiseSummary:
    """Aggregate win rate across a set of cases."""

    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0

    @property
    def total(self) -> int:
        return self.wins_a + self.wins_b + self.ties

    @property
    def win_rate_a(self) -> float:
        if self.total == 0:
            return 0.0
        # Ties count as half a win each — standard Arena / Elo convention.
        return (self.wins_a + 0.5 * self.ties) / self.total

    def add(self, result: PairwiseResult) -> None:
        if result.verdict == "a":
            self.wins_a += 1
        elif result.verdict == "b":
            self.wins_b += 1
        else:
            self.ties += 1
