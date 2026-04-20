"""Arena CLI — thin shell around the library."""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from arena import __version__
from arena.config import Settings
from arena.evals import (
    Dataset,
    RunConfig,
    RunSummary,
    VariantRunner,
)
from arena.gateway import GatewayClient
from arena.gateway.cache import SemanticCache
from arena.gateway.pricing import cost_usd
from arena.project import ProjectConfig, ProjectConfigError
from arena.store import CaseResult, JudgeScore, Run, Variant, create_engine, init_db, session
from arena.tracing import init_tracing, span

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Arena — LLM regression gym for Respan.",
)
console = Console()


def _settings() -> Settings:
    return Settings()


def _engine(settings: Settings) -> Any:
    engine = create_engine(settings.database_url)
    init_db(engine)
    return engine


def _require_respan(settings: Settings) -> None:
    if not settings.has_respan_credentials:
        console.print(
            "[red]RESPAN_API_KEY is not set.[/red] "
            "Copy .env.example to .env and fill it in, "
            "or grab a key from https://respan.ai/platform/api/api-keys."
        )
        raise typer.Exit(code=2)


def _load_project_config(path: Path = Path("arena.config.yaml")) -> ProjectConfig:
    """Load and validate `arena.config.yaml`. Returns defaults if missing."""
    try:
        return ProjectConfig.from_yaml(path)
    except ProjectConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.callback()
def _global(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    version: bool = typer.Option(False, "--version", help="Show arena version and exit."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if version:
        console.print(f"arena {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------- init

@app.command()
def init(
    project: str = typer.Argument(..., help="Directory to scaffold (will be created)."),
) -> None:
    """Create a new Arena project from the support-triage template."""
    dest = Path(project).resolve()
    if dest.exists() and any(dest.iterdir()):
        console.print(f"[red]{dest} already exists and is not empty.[/red]")
        raise typer.Exit(code=1)
    dest.mkdir(parents=True, exist_ok=True)

    (dest / ".env").write_text(
        "# Fill in from https://respan.ai/platform/api/api-keys\n"
        "RESPAN_API_KEY=\n"
        "RESPAN_ORG_ID=\n"
    )
    (dest / "arena.config.yaml").write_text(
        "dataset: dataset.jsonl\n"
        "default_model: gpt-4o-mini\n"
        "judge_model: claude-haiku-4-5\n"
        "evaluators:\n"
        "  - type: exact_match\n"
        "    field: urgency\n"
        "  - type: exact_match\n"
        "    field: category\n"
        "  - type: json_parse\n"
        "    fields: [urgency, category, suggested_reply]\n"
    )
    (dest / "prompts").mkdir(exist_ok=True)
    (dest / "prompts" / "v0.md").write_text(
        "You are a support triage assistant.\n"
        'Return a JSON object with keys "urgency", "category", "suggested_reply".\n'
    )
    (dest / "dataset.jsonl").write_text(
        json.dumps(
            {
                "id": "demo-1",
                "inputs": {"ticket": "I was charged twice for my subscription this month."},
                "expected": {"urgency": "high", "category": "billing"},
            }
        )
        + "\n"
    )
    console.print(f"[green]✓[/green] Scaffolded Arena project at {dest}")
    console.print("Next: fill in .env, then run [bold]arena run v0 --cases 1[/bold]")


# ---------------------------------------------------------------- run

_DEFAULT_DATASET = Path("dataset.jsonl")


@app.command()
def run(
    variant: str = typer.Argument(..., help="Variant name (matches prompts/<name>.md)."),
    cases: int = typer.Option(20, "--cases", help="Max cases to run."),
    dataset: Path = typer.Option(_DEFAULT_DATASET, "--dataset", help="Dataset path."),
    concurrency: int = typer.Option(4, "--concurrency", help="Max in-flight gateway calls."),
    hello_world: bool = typer.Option(
        False, "--hello-world", help="Skip dataset; make one Respan-logged call."
    ),
) -> None:
    """Run a variant across the eval set via Respan Gateway."""
    settings = _settings()
    _require_respan(settings)
    init_tracing(api_key=settings.respan_api_key_value(), app_name="arena")

    client = GatewayClient(
        api_key=settings.respan_api_key_value(),
        base_url=settings.respan_base_url,
        default_model=settings.default_model,
        disable_log=settings.disable_log,
    )

    if hello_world:
        with span("arena.run.hello_world", variant=variant):
            resp = client.chat(
                [{"role": "user", "content": "Say 'arena online' and nothing else."}],
                temperature=0.0,
                max_tokens=16,
            )
        console.print(
            f"[green]✓[/green] model={resp.model} tokens={resp.total_tokens} "
            f"latency={resp.latency_ms}ms"
        )
        console.print(f"  [dim]{resp.content!r}[/dim]")
        console.print("Check your Respan dashboard — this call is logged there.")
        return

    prompt_path = Path("prompts") / f"{variant}.md"
    if not prompt_path.exists():
        console.print(f"[red]Missing prompt file: {prompt_path}[/red]")
        raise typer.Exit(code=1)
    if not dataset.exists():
        console.print(f"[red]Missing dataset: {dataset}[/red]")
        raise typer.Exit(code=1)

    project_cfg = _load_project_config()
    ds = Dataset.from_jsonl(dataset)
    evaluators = project_cfg.to_evaluators(client=client)
    if not evaluators:
        console.print(
            "[yellow]warning[/yellow] no evaluators configured in arena.config.yaml; "
            "run will record outputs but skip scoring."
        )

    model = project_cfg.default_model or settings.default_model
    variant_row = Variant(
        name=variant,
        prompt=prompt_path.read_text(),
        model=model,
    )

    engine = _engine(settings)
    runner = VariantRunner(client=client, engine=engine)
    cache = SemanticCache(enable_semantic=True)

    summary = runner.run(
        ds,
        RunConfig(
            variant=variant_row,
            evaluators=evaluators,
            max_cases=cases,
            max_concurrency=concurrency,
            cache=cache,
        ),
    )

    _print_summary(summary)


def _print_summary(summary: RunSummary) -> None:
    scores_table = Table("evaluator", "score", title=f"run {summary.run_id} — {summary.variant_name}")
    for name, score in summary.per_evaluator.items():
        colour = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
        scores_table.add_row(name, f"[{colour}]{score:.3f}[/{colour}]")
    console.print(scores_table)

    meta = Table.grid(padding=(0, 2))
    meta.add_row("cases", f"{summary.completed_cases} / {summary.total_cases}")
    meta.add_row("errors", f"{summary.errors}")
    meta.add_row("cache hits", f"{summary.cache_hits}")
    meta.add_row(
        "tokens",
        f"in={summary.total_input_tokens}  out={summary.total_output_tokens}",
    )
    meta.add_row("cost", f"${summary.total_cost_usd:.4f}")
    console.print(meta)
    console.print(f"Run id: [bold]{summary.run_id}[/bold]")


# ---------------------------------------------------------------- placeholders

# ---------------------------------------------------------------- runs / show


@app.command("runs")
def runs(limit: int = typer.Option(20, "--limit", help="How many runs to list.")) -> None:
    """List recent runs (most recent first)."""
    from sqlmodel import desc, select

    settings = _settings()
    engine = _engine(settings)

    table = Table(
        "run id",
        "variant",
        "dataset",
        "cases",
        "status",
        "cost",
        "started",
        title="recent runs",
    )
    with session(engine) as s:
        rows = s.exec(select(Run).order_by(desc(Run.started_at)).limit(limit)).all()
        for run in rows:
            variant = s.get(Variant, run.variant_id)
            cost = _run_cost_usd(s, run.id)
            table.add_row(
                run.id[:8],
                variant.name if variant else "?",
                run.dataset,
                f"{run.completed_cases}/{run.total_cases}",
                _colour_status(run.status),
                f"${cost:.4f}",
                run.started_at.strftime("%Y-%m-%d %H:%M"),
            )

    console.print(table)


@app.command()
def show(run_id: str = typer.Argument(..., help="Run id (full or 8-char prefix).")) -> None:
    """Show the full detail of a single run — scores, costs, per-case results."""
    from sqlmodel import select

    settings = _settings()
    engine = _engine(settings)

    with session(engine) as s:
        run = _find_run(s, run_id)
        variant = s.get(Variant, run.variant_id)
        results = s.exec(
            select(CaseResult).where(CaseResult.run_id == run.id)
        ).all()
        result_ids = [r.id for r in results]
        scores = (
            s.exec(
                select(JudgeScore).where(JudgeScore.result_id.in_(result_ids))  # type: ignore[attr-defined]
            ).all()
            if result_ids
            else []
        )

    header = Table.grid(padding=(0, 2))
    header.add_row("run id", run.id)
    header.add_row("variant", variant.name if variant else "?")
    header.add_row("dataset", run.dataset)
    header.add_row("status", _colour_status(run.status))
    header.add_row("cases", f"{run.completed_cases}/{run.total_cases}")
    header.add_row(
        "tokens",
        f"in={sum(r.input_tokens for r in results)}  out={sum(r.output_tokens for r in results)}",
    )
    total_cost = sum(
        cost_usd(r.model, r.input_tokens, r.output_tokens)
        for r in results
        if not r.cache_hit
    )
    header.add_row("cost", f"${total_cost:.4f}")
    console.print(header)

    if scores:
        agg: dict[str, list[float]] = {}
        for score in scores:
            agg.setdefault(score.judge, []).append(score.score)
        scores_table = Table("evaluator", "mean score", title="scores")
        for name, values in sorted(agg.items()):
            mean = sum(values) / len(values) if values else 0.0
            colour = "green" if mean >= 0.8 else "yellow" if mean >= 0.5 else "red"
            scores_table.add_row(name, f"[{colour}]{mean:.3f}[/{colour}]")
        console.print(scores_table)

    cases_table = Table("case", "model", "tokens", "latency", "cached", title="cases")
    for r in results[:50]:
        cases_table.add_row(
            r.case_id[:8],
            r.model,
            f"{r.input_tokens}+{r.output_tokens}",
            f"{r.latency_ms}ms",
            "yes" if r.cache_hit else "",
        )
    if len(results) > 50:
        cases_table.caption = f"(showing 50 of {len(results)})"
    console.print(cases_table)


def _find_run(s: Any, run_id: str) -> Run:
    from sqlmodel import select

    run = s.get(Run, run_id)
    if run is not None:
        return run
    # prefix lookup
    candidates = s.exec(select(Run).where(Run.id.like(f"{run_id}%"))).all()  # type: ignore[attr-defined]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        console.print(f"[red]no run matching {run_id!r}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[red]ambiguous run id {run_id!r} — {len(candidates)} matches[/red]"
    )
    raise typer.Exit(code=1)


def _run_cost_usd(s: Any, run_id: str) -> float:
    from sqlmodel import select

    rows = s.exec(select(CaseResult).where(CaseResult.run_id == run_id)).all()
    return sum(
        cost_usd(r.model, r.input_tokens, r.output_tokens) for r in rows if not r.cache_hit
    )


def _colour_status(status: str) -> str:
    palette = {"done": "green", "error": "red", "running": "yellow", "pending": "dim"}
    colour = palette.get(status, "white")
    return f"[{colour}]{status}[/{colour}]"


# ---------------------------------------------------------------- placeholders


@app.command()
def judge(
    run_id: str = typer.Argument(None, help="Run id to judge (full or 8-char prefix)."),
    pairwise: bool = typer.Option(
        False, "--pairwise", help="Compare two runs instead of scoring one."
    ),
    run_b: str = typer.Option(
        None, "--vs", help="Second run id when --pairwise is set.", show_default=False
    ),
    criterion: str = typer.Option(
        "Which output is more helpful to the customer?",
        "--criterion",
        help="Pairwise criterion.",
    ),
) -> None:
    """Score a run's stored outputs with the configured judges.

    Two modes:
      arena judge <run-id>                  — apply config judges post-hoc.
      arena judge <run-a> --pairwise --vs <run-b>  — pairwise win rate.
    """
    from sqlmodel import select

    from arena.store import Case

    if run_id is None:
        console.print("[red]run-id is required[/red]")
        raise typer.Exit(code=2)

    settings = _settings()
    _require_respan(settings)
    init_tracing(api_key=settings.respan_api_key_value(), app_name="arena")

    client = GatewayClient(
        api_key=settings.respan_api_key_value(),
        base_url=settings.respan_base_url,
        default_model=settings.default_model,
        disable_log=settings.disable_log,
    )
    engine = _engine(settings)

    if pairwise:
        if run_b is None:
            console.print("[red]--pairwise requires --vs <run-id>[/red]")
            raise typer.Exit(code=2)
        _judge_pairwise(
            engine, client, run_id, run_b, criterion=criterion, model=settings.judge_model
        )
        return

    project_cfg = _load_project_config()
    judges_specs = [
        spec for spec in project_cfg.evaluators if _is_judge_spec(spec)
    ]
    if not judges_specs:
        console.print(
            "[yellow]no judge-type evaluators in arena.config.yaml[/yellow]"
        )
        return

    with session(engine) as s:
        run = _find_run(s, run_id)
        results = s.exec(select(CaseResult).where(CaseResult.run_id == run.id)).all()
        if not results:
            console.print(f"[red]run {run.id} has no persisted results[/red]")
            raise typer.Exit(code=1)

        scores_written = 0
        for result in results:
            case_row = s.get(Case, result.case_id)
            if case_row is None:
                continue
            eval_case = _case_from_row(case_row)
            for spec in judges_specs:
                judge_obj = spec.build_judge(default_model=project_cfg.judge_model)
                verdict = judge_obj.judge(eval_case, result.output, client=client)
                s.add(
                    JudgeScore(
                        result_id=result.id,
                        judge=spec.name or judge_obj.name,
                        score=verdict.score,
                        rationale=verdict.rationale,
                        raw_json=None,
                    )
                )
                scores_written += 1
        s.commit()

    console.print(
        f"[green]✓[/green] wrote {scores_written} judge scores for run "
        f"[bold]{run_id}[/bold]"
    )


def _is_judge_spec(spec: Any) -> bool:
    from arena.project import JudgeSpec

    return isinstance(spec, JudgeSpec)


def _case_from_row(row: Any) -> Any:
    from arena.evals.dataset import EvalCase

    return EvalCase(
        id=row.id,
        inputs=row.inputs,
        expected=row.expected,
        tags=row.tags,
        source=row.source,
        trace_id=row.trace_id,
    )


def _judge_pairwise(
    engine: Any,
    client: Any,
    run_a_id: str,
    run_b_id: str,
    *,
    criterion: str,
    model: str,
) -> None:
    from sqlmodel import select

    from arena.evals.dataset import EvalCase
    from arena.judges.pairwise import PairwiseJudge, PairwiseSummary
    from arena.store import Case

    judge = PairwiseJudge(criterion=criterion, model=model)
    summary = PairwiseSummary()

    with session(engine) as s:
        run_a = _find_run(s, run_a_id)
        run_b = _find_run(s, run_b_id)
        results_a = {
            r.case_id: r
            for r in s.exec(select(CaseResult).where(CaseResult.run_id == run_a.id)).all()
        }
        results_b = {
            r.case_id: r
            for r in s.exec(select(CaseResult).where(CaseResult.run_id == run_b.id)).all()
        }
        shared = sorted(set(results_a) & set(results_b))
        if not shared:
            console.print("[red]no overlapping cases between the two runs[/red]")
            raise typer.Exit(code=1)

        table = Table("case", "verdict", "rationale", title="pairwise verdicts")
        for case_id in shared:
            case_row = s.get(Case, case_id)
            if case_row is None:
                continue
            eval_case = EvalCase(
                id=case_row.id,
                inputs=case_row.inputs,
                expected=case_row.expected,
            )
            result = judge.compare(
                eval_case,
                output_a=results_a[case_id].output,
                output_b=results_b[case_id].output,
                client=client,
            )
            summary.add(result)
            table.add_row(case_id[:8], result.verdict, (result.rationale or "")[:60])
        console.print(table)

    console.print(
        f"[bold]A win rate (ties = 0.5): {summary.win_rate_a:.3f}[/bold]  "
        f"wins_a={summary.wins_a}  wins_b={summary.wins_b}  ties={summary.ties}"
    )


@app.command()
def optimize(
    variant: str = typer.Argument(..., help="Starting variant (matches prompts/<name>.md)."),
    budget: int = typer.Option(10, "--budget", help="Max proposal steps."),
    target: str = typer.Option(
        "reply_quality", "--target", help="Evaluator name to optimize against."
    ),
    cases: int = typer.Option(20, "--cases", help="Cases per run."),
    concurrency: int = typer.Option(4, "--concurrency"),
    dataset: Path = typer.Option(_DEFAULT_DATASET, "--dataset"),
    output: Path = typer.Option(
        None,
        "--output",
        help="Where to write the winning prompt (default: prompts/<variant>-optimized.md).",
    ),
) -> None:
    """Auto-propose prompt variants via textual gradients.

    Starts from the baseline prompt, runs it, picks the worst cases on the
    target evaluator, asks the optimizer model for a revised prompt, runs
    the revision, keeps it if it scores higher, and repeats until the
    budget is spent. The best prompt is written back to disk so it can be
    re-run with ``arena run``.
    """
    from arena.optimizer import OptimizerConfig
    from arena.optimizer import optimize as run_optimizer

    settings = _settings()
    _require_respan(settings)
    init_tracing(api_key=settings.respan_api_key_value(), app_name="arena")

    prompt_path = Path("prompts") / f"{variant}.md"
    if not prompt_path.exists():
        console.print(f"[red]Missing prompt file: {prompt_path}[/red]")
        raise typer.Exit(code=1)
    if not dataset.exists():
        console.print(f"[red]Missing dataset: {dataset}[/red]")
        raise typer.Exit(code=1)

    client = GatewayClient(
        api_key=settings.respan_api_key_value(),
        base_url=settings.respan_base_url,
        default_model=settings.default_model,
        disable_log=settings.disable_log,
    )
    project_cfg = _load_project_config()
    ds = Dataset.from_jsonl(dataset)
    evaluators = project_cfg.to_evaluators(client=client)
    if not any(e.name == target for e in evaluators):
        available = ", ".join(e.name for e in evaluators) or "(none)"
        console.print(
            f"[red]target evaluator {target!r} not found. Available: {available}[/red]"
        )
        raise typer.Exit(code=2)

    parent = Variant(
        name=variant,
        prompt=prompt_path.read_text(),
        model=project_cfg.default_model or settings.default_model,
    )
    engine = _engine(settings)
    optimizer_model = project_cfg.optimizer_model or settings.optimizer_model

    with span("arena.optimize", variant=variant, budget=budget, target=target):
        report = run_optimizer(
            OptimizerConfig(
                parent_variant=parent,
                dataset=ds,
                target_evaluator=target,
                evaluators=evaluators,
                budget=budget,
                max_cases=cases,
                concurrency=concurrency,
                optimizer_model=optimizer_model,
            ),
            client=client,
            engine=engine,
        )

    _print_optimizer_report(report)

    # Write the winning prompt back so ``arena run`` can pick it up next.
    output_path = output or (Path("prompts") / f"{variant}-optimized.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.best_variant.prompt)
    console.print(
        f"[green]✓[/green] wrote best prompt to [bold]{output_path}[/bold]"
    )


def _print_optimizer_report(report: Any) -> None:
    steps_table = Table(
        "step",
        "kept",
        "score before",
        "score after",
        "Δ",
        "gradient",
        title="optimizer steps",
    )
    for step in report.steps:
        delta = step.score_after - step.score_before
        colour = "green" if step.kept else "dim"
        steps_table.add_row(
            str(step.step),
            f"[{colour}]{'yes' if step.kept else 'no'}[/{colour}]",
            f"{step.score_before:.3f}",
            f"{step.score_after:.3f}",
            f"{delta:+.3f}",
            (step.gradient or "")[:60],
        )
    console.print(steps_table)

    summary = Table.grid(padding=(0, 2))
    summary.add_row("parent score", f"{report.parent_score:.3f}")
    summary.add_row("best score", f"[bold]{report.best_score:.3f}[/bold]")
    summary.add_row("best run id", report.best_run_id)
    summary.add_row("promoted", "yes" if report.promoted else "no")
    console.print(summary)


@app.command()
def mine(
    from_respan: bool = typer.Option(
        False, "--from-respan", help="Pull from a live Respan workspace."
    ),
    fixture: Path = typer.Option(
        None, "--fixture", help="Load traces from a local JSONL fixture instead.",
    ),
    last: str = typer.Option("24h", "--last", help="Lookback window (e.g. 30m, 24h, 7d)."),
    limit: int = typer.Option(500, "--limit", help="Max traces to pull."),
    min_cluster: int = typer.Option(3, "--min-cluster", help="Min traces per cluster."),
    only_failures: bool = typer.Option(
        True, "--only-failures/--all", help="Only mine error/flagged traces."
    ),
    output: Path = typer.Option(
        Path("mined.jsonl"), "--output", help="Where to write the eval cases."
    ),
    no_label: bool = typer.Option(
        False, "--no-label", help="Skip LLM cluster labelling (offline mode).",
    ),
) -> None:
    """Mine Respan traces into an eval dataset.

    Two modes:
      arena mine --from-respan --last 24h           — live workspace pull.
      arena mine --fixture examples/.../traces.jsonl — offline demo mode.
    """
    from datetime import datetime

    from arena.mine import (
        FixtureSource,
        MiningReport,
        RespanAPI,
        mine_to_eval_cases,
        parse_relative_duration,
    )

    if not from_respan and fixture is None:
        console.print("[red]pass --from-respan or --fixture <path>[/red]")
        raise typer.Exit(code=2)
    if from_respan and fixture is not None:
        console.print("[red]--from-respan and --fixture are mutually exclusive[/red]")
        raise typer.Exit(code=2)

    settings = _settings()

    if from_respan:
        _require_respan(settings)
        try:
            datetime.now(tz=UTC) - parse_relative_duration(last)
        except ValueError as exc:
            console.print(f"[red]invalid --last: {exc}[/red]")
            raise typer.Exit(code=2) from exc
        # Respan traces use a non-gateway base URL in most deployments; the
        # env knob lets the user override to their workspace host.
        source = RespanAPI(
            api_key=settings.respan_api_key_value(),
            base_url=settings.respan_base_url.replace("/api", ""),
        )
        init_tracing(api_key=settings.respan_api_key_value(), app_name="arena")
        label_client = GatewayClient(
            api_key=settings.respan_api_key_value(),
            base_url=settings.respan_base_url,
            default_model=settings.default_model,
        ) if not no_label else None
    else:
        source = FixtureSource(fixture)  # type: ignore[arg-type]
        label_client = (
            GatewayClient(
                api_key=settings.respan_api_key_value(),
                base_url=settings.respan_base_url,
                default_model=settings.default_model,
            )
            if not no_label and settings.has_respan_credentials
            else None
        )

    with span("arena.mine"):
        report: MiningReport = mine_to_eval_cases(
            source,
            client=label_client,
            min_cluster_size=min_cluster,
            max_traces=limit,
            only_failures=only_failures,
            label_clusters=not no_label,
        )

    # Write eval cases to output JSONL — this is what `arena run --dataset` consumes.
    with output.open("w") as fh:
        for case in report.cases:
            fh.write(
                json.dumps(
                    {
                        "id": case.id,
                        "inputs": case.inputs,
                        "expected": case.expected,
                        "tags": case.tags,
                        "source": case.source,
                        "trace_id": case.trace_id,
                    }
                )
                + "\n"
            )

    # Print a per-cluster summary so the user can eyeball the mine.
    table = Table("cluster", "label", "size", title="mined clusters")
    for cluster in report.clusters:
        table.add_row(str(cluster.id), cluster.label or "(no label)", str(cluster.size))
    console.print(table)
    console.print(
        f"[green]✓[/green] {report.total_traces} traces → "
        f"{len(report.clusters)} clusters → {len(report.cases)} cases "
        f"written to [bold]{output}[/bold]"
    )


@app.command()
def gate(
    baseline: str = typer.Option(
        ..., "--baseline", help="Baseline run id (full or 8-char prefix)."
    ),
    run_id: str = typer.Option(
        None,
        "--run",
        help="Candidate run id. Defaults to the most recent completed run.",
    ),
    threshold: float = typer.Option(
        None,
        "--threshold",
        help="Max allowed per-judge drop (0..1). Default 0.02.",
    ),
) -> None:
    """CI regression gate — exit 1 if the candidate run regresses vs a baseline."""
    from sqlmodel import desc, select

    from arena.gate import DEFAULT_THRESHOLD, evaluate

    settings = _settings()
    engine = _engine(settings)

    with session(engine) as s:
        baseline_run = _find_run(s, baseline)
        if run_id is None:
            latest = s.exec(
                select(Run).order_by(desc(Run.started_at)).limit(1)
            ).first()
            if latest is None:
                console.print("[red]no runs in DB to gate against[/red]")
                raise typer.Exit(code=1)
            candidate_run = latest
        else:
            candidate_run = _find_run(s, run_id)

    report = evaluate(
        engine=engine,
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
        threshold=threshold if threshold is not None else DEFAULT_THRESHOLD,
    )
    _print_gate_report(report)
    if not report.passed:
        raise typer.Exit(code=1)


def _print_gate_report(report: Any) -> None:
    table = Table(
        "judge",
        "baseline",
        "candidate",
        "Δ",
        "status",
        title=f"gate — {report.candidate_run_id[:8]} vs {report.baseline_run_id[:8]}",
    )
    for delta in report.deltas:
        candidate = "missing" if delta.candidate is None else f"{delta.candidate:.3f}"
        tone = "red" if delta.regressed else "green"
        table.add_row(
            delta.judge,
            f"{delta.baseline:.3f}",
            candidate,
            f"[{tone}]{delta.delta:+.3f}[/{tone}]",
            f"[{tone}]{'REGRESS' if delta.regressed else 'ok'}[/{tone}]",
        )
    console.print(table)
    for note in report.notes:
        console.print(f"[yellow]note[/yellow] {note}")

    verdict = (
        "[green bold]PASS[/green bold]"
        if report.passed
        else "[red bold]FAIL[/red bold]"
    )
    console.print(
        f"{verdict} threshold={report.threshold:.3f} "
        f"regressions={len(report.regressed_judges)}"
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
