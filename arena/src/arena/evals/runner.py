"""Concurrent variant runner.

Given a dataset, a variant (prompt + model + temp), and a list of
evaluators, the runner fires off gateway calls in parallel, passes each
response through every evaluator, and persists a `Run` plus per-case
`CaseResult`s to the database.

Design notes:
- Concurrency is bounded by a `ThreadPoolExecutor`. The Respan Gateway is
  I/O-bound (network), so threads are the right tool — no GIL pressure.
- The semantic cache is hit *before* the gateway call and populated
  *after* it; cache hits short-circuit both the network call and the
  retry policy.
- Failures on individual cases don't kill the run. We record the
  exception on the `CaseResult` and keep going, so a flaky endpoint can't
  poison a 200-case eval.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine

from arena.evals.dataset import Dataset, EvalCase
from arena.evals.evaluators import Evaluator, EvaluatorResult
from arena.gateway.cache import SemanticCache
from arena.gateway.client import GatewayClient, GatewayError, GatewayResponse
from arena.store import CaseResult, Run, Variant, session
from arena.tracing import span

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RunConfig:
    variant: Variant
    evaluators: list[Evaluator]
    max_concurrency: int = 4
    max_cases: int | None = None
    cache: SemanticCache | None = None


@dataclass(slots=True)
class CaseOutcome:
    case: EvalCase
    output: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cache_hit: bool
    scores: list[EvaluatorResult] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores)


@dataclass(slots=True)
class RunSummary:
    run_id: str
    variant_name: str
    total_cases: int
    completed_cases: int
    errors: int
    per_evaluator: dict[str, float]
    total_input_tokens: int
    total_output_tokens: int
    cache_hits: int
    outcomes: list[CaseOutcome]

    @property
    def pass_rate(self) -> float:
        return (self.completed_cases - self.errors) / max(self.completed_cases, 1)


class VariantRunner:
    """Runs a variant across a dataset via a GatewayClient."""

    def __init__(self, client: GatewayClient, engine: Engine) -> None:
        self._client = client
        self._engine = engine

    def run(self, dataset: Dataset, cfg: RunConfig) -> RunSummary:
        cases = list(dataset.cases[: cfg.max_cases]) if cfg.max_cases else list(dataset)

        with session(self._engine) as s:
            # Persist the variant if it doesn't already have an id.
            if not cfg.variant.id or not s.get(Variant, cfg.variant.id):
                s.add(cfg.variant)
                s.commit()
                s.refresh(cfg.variant)
            run = Run(
                variant_id=cfg.variant.id,
                dataset=dataset.name,
                status="running",
                total_cases=len(cases),
            )
            s.add(run)
            s.commit()
            s.refresh(run)
            run_id = run.id

        outcomes: list[CaseOutcome] = []
        cache_hits = 0

        with (
            span("arena.run", run_id=run_id, variant=cfg.variant.name, cases=len(cases)),
            ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool,
        ):
            futures = {pool.submit(self._run_one, case, cfg): case for case in cases}
            for fut in as_completed(futures):
                outcome = fut.result()
                outcomes.append(outcome)
                if outcome.cache_hit:
                    cache_hits += 1

        self._persist_results(run_id, outcomes)
        self._finalise_run(run_id, len(cases), outcomes)

        per_evaluator = _aggregate_scores(outcomes, cfg.evaluators)
        errors = sum(1 for o in outcomes if o.error)
        return RunSummary(
            run_id=run_id,
            variant_name=cfg.variant.name,
            total_cases=len(cases),
            completed_cases=len(outcomes),
            errors=errors,
            per_evaluator=per_evaluator,
            total_input_tokens=sum(o.input_tokens for o in outcomes),
            total_output_tokens=sum(o.output_tokens for o in outcomes),
            cache_hits=cache_hits,
            outcomes=outcomes,
        )

    # ------------------------------------------------------------ internals

    def _run_one(self, case: EvalCase, cfg: RunConfig) -> CaseOutcome:
        messages = [
            {"role": "system", "content": cfg.variant.prompt},
            {"role": "user", "content": case.user_text()},
        ]
        temperature = cfg.variant.temperature

        cache_hit = False
        resp: GatewayResponse | None = None
        if cfg.cache is not None:
            cached = cfg.cache.get(messages, cfg.variant.model, temperature)
            if cached is not None:
                resp = cached
                cache_hit = True

        if resp is None:
            t0 = time.perf_counter()
            try:
                with span("arena.case", case_id=case.id):
                    resp = self._client.chat(
                        messages,
                        model=cfg.variant.model,
                        temperature=temperature,
                        max_tokens=cfg.variant.max_tokens,
                    )
            except GatewayError as exc:
                return CaseOutcome(
                    case=case,
                    output="",
                    model=cfg.variant.model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    cache_hit=False,
                    error=str(exc),
                )
            if cfg.cache is not None:
                cfg.cache.put(messages, cfg.variant.model, temperature, resp)

        scores = [e.score(case, resp.content) for e in cfg.evaluators]
        return CaseOutcome(
            case=case,
            output=resp.content,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
            cache_hit=cache_hit,
            scores=scores,
        )

    def _persist_results(self, run_id: str, outcomes: list[CaseOutcome]) -> None:
        with session(self._engine) as s:
            for outcome in outcomes:
                s.add(
                    CaseResult(
                        run_id=run_id,
                        case_id=outcome.case.id,
                        output=outcome.output,
                        input_tokens=outcome.input_tokens,
                        output_tokens=outcome.output_tokens,
                        latency_ms=outcome.latency_ms,
                        model=outcome.model,
                        cache_hit=outcome.cache_hit,
                        error=outcome.error,
                    )
                )
            s.commit()

    def _finalise_run(
        self, run_id: str, total: int, outcomes: list[CaseOutcome]
    ) -> None:
        with session(self._engine) as s:
            run = s.get(Run, run_id)
            if run is None:
                return
            run.status = "done" if all(o.error is None for o in outcomes) else "error"
            run.completed_cases = len(outcomes)
            s.add(run)
            s.commit()


def _aggregate_scores(
    outcomes: list[CaseOutcome],
    evaluators: list[Evaluator],
) -> dict[str, float]:
    agg: dict[str, list[float]] = {e.name: [] for e in evaluators}
    for outcome in outcomes:
        for score in outcome.scores:
            agg.setdefault(score.name, []).append(score.score)
    return {name: (sum(v) / len(v) if v else 0.0) for name, v in agg.items()}


def _default_evaluators_from_config(cfg: dict[str, Any]) -> list[Evaluator]:
    """Build the evaluator list from a parsed arena.config.yaml section."""
    from arena.evals.evaluators import (
        ExactMatchEvaluator,
        JSONParseEvaluator,
        RegexEvaluator,
    )

    out: list[Evaluator] = []
    for spec in cfg.get("evaluators", []) or []:
        kind = spec.get("type")
        if kind == "exact_match":
            out.append(ExactMatchEvaluator(field=spec["field"]))
        elif kind == "json_parse":
            out.append(JSONParseEvaluator(required_fields=tuple(spec["fields"])))
        elif kind == "regex":
            out.append(RegexEvaluator(pattern=spec["pattern"]))
        elif kind == "judge":
            # Added in Week 1 Day 5 — skipped for now with a clear log.
            log.info("skipping judge evaluator %s (wired in Day 5)", spec)
        else:
            log.warning("unknown evaluator type: %s", kind)
    return out
