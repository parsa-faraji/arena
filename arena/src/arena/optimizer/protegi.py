"""Textual-gradient prompt optimizer (ProTeGi-style).

Strategy per step:
  1. Take the current best variant's per-case scores on the target evaluator.
  2. Pick the K lowest-scoring cases — these are the "failure examples".
  3. Ask the optimizer LLM: "Here's the prompt, here's what failed, here's
     why. Propose a revised prompt that keeps the wins but fixes these."
  4. Persist the proposal as a new Variant, run it on the same cases,
     score it with the same evaluators.
  5. If the mean target score beats the parent, promote — the proposal
     becomes the new best. Either way, record an OptimizerStep row.

The loop stops once `budget` proposal calls have been issued. Each step
costs one proposal call plus one full run (N gateway calls + judges).

We deliberately stay close to the ProTeGi paper's textual-gradient idea
without implementing the beam search / paraphrase variants — the cost/
complexity tradeoff isn't worth it for a portfolio demo, and the simple
greedy climb already beats v0 reliably on the support-triage example.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlmodel import select

from arena.evals.dataset import Dataset
from arena.evals.evaluators import Evaluator
from arena.evals.runner import CaseOutcome, RunConfig, VariantRunner
from arena.gateway.client import GatewayClient
from arena.judges.base import JUDGE_SYSTEM_PROMPT
from arena.store import CaseResult, JudgeScore, OptimizerStep, Variant, session

if TYPE_CHECKING:
    from sqlalchemy import Engine

log = logging.getLogger(__name__)


OPTIMIZER_SYSTEM_PROMPT = (
    "You are a prompt engineer. You'll be shown a system prompt that is "
    "underperforming on a named criterion, plus a handful of concrete "
    "failures. Propose an improved system prompt that keeps the existing "
    "behaviour but fixes the named failures. Respond ONLY with a JSON "
    'object of shape {"prompt": "<the full revised system prompt>", '
    '"gradient": "<one short sentence naming the fix>"}. '
    "No prose, no code fences."
)


@dataclass(slots=True)
class OptimizerConfig:
    parent_variant: Variant
    dataset: Dataset
    target_evaluator: str
    evaluators: list[Evaluator]
    budget: int = 10
    max_cases: int = 20
    concurrency: int = 4
    failures_per_step: int = 3
    optimizer_model: str = "claude-sonnet-4-6"
    optimizer_temperature: float = 0.4


@dataclass(slots=True)
class StepRecord:
    step: int
    proposed_variant_id: str
    proposed_prompt: str
    gradient: str
    score_before: float
    score_after: float
    kept: bool


@dataclass(slots=True)
class OptimizerReport:
    parent_run_id: str
    parent_score: float
    best_variant: Variant
    best_run_id: str
    best_score: float
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.best_variant.id != self._parent_variant_id

    # Filled in by the optimizer before returning.
    _parent_variant_id: str = ""


def optimize(
    cfg: OptimizerConfig,
    *,
    client: GatewayClient,
    engine: Engine,
) -> OptimizerReport:
    """Run the optimizer loop. Returns a report summarising every step."""
    runner = VariantRunner(client=client, engine=engine)

    parent_summary = runner.run(
        cfg.dataset,
        RunConfig(
            variant=cfg.parent_variant,
            evaluators=cfg.evaluators,
            max_cases=cfg.max_cases,
            max_concurrency=cfg.concurrency,
        ),
    )
    parent_score = parent_summary.per_evaluator.get(cfg.target_evaluator, 0.0)

    best_variant = cfg.parent_variant
    best_summary = parent_summary
    best_score = parent_score
    tried_prompts: set[str] = {cfg.parent_variant.prompt.strip()}
    steps: list[StepRecord] = []

    for step_idx in range(1, cfg.budget + 1):
        failures = _pick_failures(
            best_summary.outcomes, cfg.target_evaluator, k=cfg.failures_per_step
        )
        proposal = _propose_prompt(
            client=client,
            parent_prompt=best_variant.prompt,
            criterion=cfg.target_evaluator,
            failures=failures,
            model=cfg.optimizer_model,
            temperature=cfg.optimizer_temperature,
        )
        if proposal is None or proposal.prompt.strip() in tried_prompts:
            log.info("optimize step %d: no usable proposal, skipping", step_idx)
            continue
        tried_prompts.add(proposal.prompt.strip())

        proposed_variant = Variant(
            name=f"{cfg.parent_variant.name}-opt-{step_idx}",
            prompt=proposal.prompt,
            model=cfg.parent_variant.model,
            temperature=cfg.parent_variant.temperature,
            max_tokens=cfg.parent_variant.max_tokens,
            notes=f"optimizer step {step_idx} from {cfg.parent_variant.id[:8]}",
        )
        proposal_summary = runner.run(
            cfg.dataset,
            RunConfig(
                variant=proposed_variant,
                evaluators=cfg.evaluators,
                max_cases=cfg.max_cases,
                max_concurrency=cfg.concurrency,
            ),
        )
        proposal_score = proposal_summary.per_evaluator.get(cfg.target_evaluator, 0.0)
        kept = proposal_score > best_score

        _persist_step(
            engine=engine,
            parent_id=best_variant.id,
            proposed_id=proposed_variant.id,
            step=step_idx,
            gradient=proposal.gradient,
            score_before=best_score,
            score_after=proposal_score,
            kept=kept,
        )

        steps.append(
            StepRecord(
                step=step_idx,
                proposed_variant_id=proposed_variant.id,
                proposed_prompt=proposal.prompt,
                gradient=proposal.gradient,
                score_before=best_score,
                score_after=proposal_score,
                kept=kept,
            )
        )

        if kept:
            best_variant = proposed_variant
            best_summary = proposal_summary
            best_score = proposal_score

    report = OptimizerReport(
        parent_run_id=parent_summary.run_id,
        parent_score=parent_score,
        best_variant=best_variant,
        best_run_id=best_summary.run_id,
        best_score=best_score,
        steps=steps,
    )
    report._parent_variant_id = cfg.parent_variant.id
    return report


# ------------------------------------------------------------ internals


@dataclass(slots=True)
class _Proposal:
    prompt: str
    gradient: str


def _propose_prompt(
    *,
    client: GatewayClient,
    parent_prompt: str,
    criterion: str,
    failures: list[tuple[str, str, float, str]],
    model: str,
    temperature: float,
) -> _Proposal | None:
    """Ask the optimizer LLM for a revised prompt.

    failures: list of (case_input, model_output, score, rationale).
    """
    if not failures:
        # No signal — skip the proposal rather than waste a budget unit.
        return None

    bullets = []
    for i, (user_input, output, score, rationale) in enumerate(failures, start=1):
        bullets.append(
            f"Example {i} (score={score:.2f}):\n"
            f"  input: {user_input}\n"
            f"  model output: {output}\n"
            f"  judge note: {rationale or '(no rationale)'}"
        )

    user = (
        f"Current system prompt:\n---\n{parent_prompt}\n---\n\n"
        f"The prompt is being judged on: {criterion}\n\n"
        "Here are specific failures where the prompt underperformed:\n\n"
        + "\n\n".join(bullets)
        + "\n\nPropose a revised system prompt."
    )
    resp = client.chat(
        [
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    parsed = _parse_proposal(resp.content)
    if parsed is None:
        log.warning("optimize: could not parse proposal: %r", resp.content[:200])
        return None
    return parsed


def _parse_proposal(content: str) -> _Proposal | None:
    text = content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Sometimes models wrap JSON in fences even when told not to.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    gradient = str(data.get("gradient", "")).strip() or "(no gradient)"
    return _Proposal(prompt=prompt.strip(), gradient=gradient)


def _pick_failures(
    outcomes: list[CaseOutcome],
    evaluator_name: str,
    *,
    k: int,
) -> list[tuple[str, str, float, str]]:
    """Sort cases by the named evaluator's score, ascending, take the worst k."""
    scored: list[tuple[CaseOutcome, float, str]] = []
    for outcome in outcomes:
        hit = next((s for s in outcome.scores if s.name == evaluator_name), None)
        if hit is None:
            continue
        rationale = ""
        if hit.details:
            rationale = str(hit.details.get("rationale", ""))
        scored.append((outcome, hit.score, rationale))
    scored.sort(key=lambda r: r[1])
    worst = scored[:k]
    return [
        (o.case.user_text(), o.output[:400], score, rationale)
        for (o, score, rationale) in worst
    ]


def _persist_step(
    *,
    engine: Engine,
    parent_id: str,
    proposed_id: str,
    step: int,
    gradient: str,
    score_before: float,
    score_after: float,
    kept: bool,
) -> None:
    with session(engine) as s:
        s.add(
            OptimizerStep(
                parent_variant_id=parent_id,
                proposed_variant_id=proposed_id,
                step=step,
                gradient_text=gradient,
                score_before=score_before,
                score_after=score_after,
                kept=kept,
            )
        )
        s.commit()


def latest_judge_score(
    engine: Engine, run_id: str, judge_name: str
) -> float | None:
    """Convenience: average score for a named judge on a run. For callers
    that don't have the RunSummary in memory (e.g. the CLI gate command)."""
    with session(engine) as s:
        results = s.exec(
            select(CaseResult.id).where(CaseResult.run_id == run_id)
        ).all()
        if not results:
            return None
        result_ids = list(results)
        scores = s.exec(
            select(JudgeScore.score).where(
                (JudgeScore.result_id.in_(result_ids))  # type: ignore[attr-defined]
                & (JudgeScore.judge == judge_name)
            )
        ).all()
    if not scores:
        return None
    return sum(scores) / len(scores)


# Keep the import surface referenced by tests stable.
__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "OPTIMIZER_SYSTEM_PROMPT",
    "OptimizerConfig",
    "OptimizerReport",
    "StepRecord",
    "latest_judge_score",
    "optimize",
]
