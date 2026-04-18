from __future__ import annotations

from dataclasses import dataclass

from arena.evals.dataset import EvalCase
from arena.judges.base import JudgeVerdict
from arena.judges.ensemble import EnsembleAgreement, JudgeEnsemble


@dataclass
class _FakeJudge:
    """Returns a fixed verdict regardless of input — good enough for testing."""

    name: str
    score: float
    rationale: str = ""

    def judge(self, case, output, *, client):  # noqa: ANN001
        return JudgeVerdict(score=self.score, rationale=self.rationale)


def _case() -> EvalCase:
    return EvalCase(id="c1", inputs={"ticket": "t"})


def test_ensemble_mean_and_agreement_when_judges_agree() -> None:
    ens = JudgeEnsemble(
        judges=[_FakeJudge("a", 0.8), _FakeJudge("b", 0.8), _FakeJudge("c", 0.8)],
    )
    v = ens.judge(_case(), "anything", client=None)  # type: ignore[arg-type]
    assert v.score == 0.8
    assert v.raw["agreement"] == 1.0
    assert v.raw["stdev"] == 0.0


def test_ensemble_agreement_shrinks_when_judges_disagree() -> None:
    ens = JudgeEnsemble(judges=[_FakeJudge("a", 0.0), _FakeJudge("b", 1.0)])
    v = ens.judge(_case(), "x", client=None)  # type: ignore[arg-type]
    assert v.score == 0.5
    assert v.raw["agreement"] == 0.0  # maximal disagreement
    assert len(v.raw["judges"]) == 2


def test_ensemble_rationale_is_central_judge() -> None:
    ens = JudgeEnsemble(
        judges=[
            _FakeJudge("low", 0.1, rationale="bad"),
            _FakeJudge("mid", 0.5, rationale="middling"),
            _FakeJudge("hi", 0.9, rationale="great"),
        ]
    )
    v = ens.judge(_case(), "x", client=None)  # type: ignore[arg-type]
    assert v.rationale == "middling"


def test_ensemble_rejects_empty_judge_list() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one judge"):
        JudgeEnsemble(judges=[])


def test_run_level_agreement_aggregation() -> None:
    verdicts = {
        "c1": JudgeVerdict(score=0.8, rationale="", raw={"agreement": 1.0}),
        "c2": JudgeVerdict(score=0.5, rationale="", raw={"agreement": 0.4}),
        "c3": JudgeVerdict(score=0.9, rationale="", raw={"agreement": 0.9}),
    }
    stats = EnsembleAgreement.from_verdicts(verdicts, threshold=0.6)
    assert stats.cases == 3
    assert stats.mean_agreement == (1.0 + 0.4 + 0.9) / 3
    assert stats.low_agreement_cases == ["c2"]
