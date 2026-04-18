"""Judge ensemble — multiple judges for the same (case, output) pair.

Why: single judges (especially small models) are noisy. Running two or
three different judges and averaging their scores produces a steadier
signal AND surfaces *disagreement*, which is the real gold — cases where
the judges disagree are the ones worth looking at by hand.

The ensemble returns a single score (the mean) and records the full
distribution plus an `agreement` metric in [0, 1] on the raw dict so
the dashboard and `arena show` can display it. Agreement = 1 - stdev(s),
clamped to [0, 1]. With scores in [0, 1], the max stdev is 0.5 (e.g.
one judge says 0, the other says 1), so agreement collapses to 0 only
when judges are maximally split.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from arena.evals.dataset import EvalCase
from arena.gateway.client import GatewayClient
from arena.judges.base import Judge, JudgeVerdict


@dataclass
class JudgeEnsemble:
    """Runs multiple judges and averages their scores.

    Agreement is exposed as `raw["agreement"]` per verdict so the runner
    can persist it. It's also logged so low-agreement cases stand out in
    the CLI output.
    """

    judges: list[Judge]
    name: str = "ensemble"
    max_concurrency: int = 3

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("JudgeEnsemble needs at least one judge")

    def judge(
        self,
        case: EvalCase,
        output: str,
        *,
        client: GatewayClient,
    ) -> JudgeVerdict:
        verdicts = self._run_all(case, output, client)
        scores = [v.score for v in verdicts]
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        agreement = max(0.0, min(1.0, 1.0 - 2 * stdev))

        per_judge = [
            {"judge": j.name, "score": v.score, "rationale": v.rationale}
            for j, v in zip(self.judges, verdicts, strict=True)
        ]
        rationale = _pick_representative_rationale(verdicts)
        return JudgeVerdict(
            score=mean,
            rationale=rationale,
            raw={
                "mean": mean,
                "stdev": stdev,
                "agreement": agreement,
                "judges": per_judge,
            },
        )

    # ------------------------------------------------------------ helpers

    def _run_all(
        self,
        case: EvalCase,
        output: str,
        client: GatewayClient,
    ) -> list[JudgeVerdict]:
        if len(self.judges) == 1:
            return [self.judges[0].judge(case, output, client=client)]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(self.judges))) as pool:
            futures = [
                pool.submit(j.judge, case, output, client=client) for j in self.judges
            ]
            return [f.result() for f in futures]


def _pick_representative_rationale(verdicts: list[JudgeVerdict]) -> str:
    """Rationale of the judge closest to the ensemble mean — most central view."""
    if not verdicts:
        return ""
    mean = statistics.mean(v.score for v in verdicts)
    closest = min(verdicts, key=lambda v: abs(v.score - mean))
    return closest.rationale


@dataclass
class EnsembleAgreement:
    """Run-level agreement stats across many ensemble verdicts."""

    cases: int
    mean_agreement: float
    low_agreement_cases: list[str] = field(default_factory=list)

    @classmethod
    def from_verdicts(
        cls,
        verdicts_by_case: dict[str, JudgeVerdict],
        *,
        threshold: float = 0.6,
    ) -> EnsembleAgreement:
        agreements = [
            float(v.raw.get("agreement", 1.0)) for v in verdicts_by_case.values()
        ]
        low = [
            cid
            for cid, v in verdicts_by_case.items()
            if float(v.raw.get("agreement", 1.0)) < threshold
        ]
        return cls(
            cases=len(agreements),
            mean_agreement=(sum(agreements) / len(agreements)) if agreements else 1.0,
            low_agreement_cases=low,
        )
