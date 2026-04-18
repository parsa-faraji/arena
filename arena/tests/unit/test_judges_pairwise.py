from __future__ import annotations

import random
from typing import Any

from arena.evals.dataset import EvalCase
from arena.gateway.client import GatewayResponse
from arena.judges.pairwise import PairwiseJudge, PairwiseSummary


class _FakeClient:
    def __init__(self, picks: list[str]) -> None:
        """`picks` are per-call replies: '1', '2', or 'tie'."""
        self._picks = picks
        self.i = 0
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> GatewayResponse:
        self.prompts.append(messages[-1]["content"])
        pick = self._picks[self.i]
        self.i += 1
        return GatewayResponse(
            content=f'{{"winner": "{pick}", "rationale": "x"}}',
            model="t", input_tokens=1, output_tokens=1, latency_ms=1, raw={},
        )


def test_pairwise_verdict_mapped_back_to_a_or_b() -> None:
    # Force swap=False (rng returns 0.9) so output 1 == A, output 2 == B.
    judge = PairwiseJudge(criterion="helpful", rng=random.Random(999))
    client = _FakeClient(["1"])
    result = judge.compare(
        EvalCase(id="c", inputs={"ticket": "t"}),
        output_a="A",
        output_b="B",
        client=client,  # type: ignore[arg-type]
    )
    assert result.verdict in {"a", "b"}
    # Whoever was presented first won.
    assert result.verdict == result.presented_first


def test_pairwise_swap_flips_mapping() -> None:
    # rng.random() < 0.5 triggers a swap.
    # Construct a deterministic rng that returns 0.1 first.
    judge_swap = PairwiseJudge(criterion="h", rng=_SeqRandom([0.1]))
    client = _FakeClient(["1"])  # model picks "output 1"
    result = judge_swap.compare(
        EvalCase(id="c", inputs={"ticket": "t"}),
        output_a="A",
        output_b="B",
        client=client,  # type: ignore[arg-type]
    )
    # When swapped, output 1 is B, so the model "picking 1" is a B win.
    assert result.verdict == "b"
    assert result.presented_first == "b"


def test_pairwise_tie_passthrough() -> None:
    judge = PairwiseJudge(criterion="h", rng=random.Random(0))
    client = _FakeClient(["tie"])
    result = judge.compare(
        EvalCase(id="c", inputs={"ticket": "t"}),
        output_a="A", output_b="B", client=client,  # type: ignore[arg-type]
    )
    assert result.verdict == "tie"


def test_summary_win_rate_with_ties() -> None:
    summary = PairwiseSummary()
    judge = PairwiseJudge(criterion="h", rng=random.Random(0))
    client = _FakeClient(["1", "2", "tie", "1"])
    for _ in range(4):
        summary.add(
            judge.compare(
                EvalCase(id="c", inputs={}),
                output_a="A", output_b="B", client=client,  # type: ignore[arg-type]
            )
        )
    # Exact counts depend on rng-driven swaps, but total should be 4.
    assert summary.total == 4
    assert summary.wins_a + summary.wins_b + summary.ties == 4


class _SeqRandom(random.Random):
    """Random subclass that replays a fixed sequence via `random()`."""

    def __init__(self, seq: list[float]) -> None:
        super().__init__()
        self._seq = list(seq)
        self._i = 0

    def random(self) -> float:  # type: ignore[override]
        v = self._seq[self._i % len(self._seq)]
        self._i += 1
        return v
