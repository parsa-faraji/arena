"""Arena CLI — thin shell around the library."""
from __future__ import annotations

import json
import logging
import sys
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
    evaluators = project_cfg.to_evaluators()
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
def judge(run_id: str = typer.Argument(...)) -> None:
    """Score a run with the configured judge ensemble. (Week-1 stub.)"""
    console.print(f"[yellow]TODO[/yellow] judge run {run_id} - Week 1 Day 5.")


@app.command()
def optimize(
    variant: str = typer.Argument(...),
    budget: int = typer.Option(30, "--budget"),
    target: str = typer.Option("reply_quality", "--target"),
) -> None:
    """Auto-propose prompt variants. (Week-2 stub.)"""
    console.print(
        f"[yellow]TODO[/yellow] optimize {variant} budget={budget} target={target} - Week 2 Day 8."
    )


@app.command()
def mine(
    from_respan: bool = typer.Option(False, "--from-respan"),
    last: str = typer.Option("24h", "--last"),
) -> None:
    """Pull + cluster Respan traces into eval cases. (Week-1 stub.)"""
    console.print(
        f"[yellow]TODO[/yellow] mine from_respan={from_respan} last={last} - Week 1 Day 6-7."
    )


@app.command()
def gate(baseline: str = typer.Option(..., "--baseline")) -> None:
    """CI regression gate — exit 1 if current run regresses. (Week-2 stub.)"""
    console.print(f"[yellow]TODO[/yellow] gate vs baseline={baseline} - Week 2 Day 10.")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
