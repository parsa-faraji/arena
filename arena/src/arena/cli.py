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
from arena.gateway import GatewayClient
from arena.store import Run, Variant, create_engine, init_db, session
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
        "judges: [pairwise, rubric]\n"
    )
    (dest / "prompts").mkdir(exist_ok=True)
    (dest / "prompts" / "v0-baseline.md").write_text(
        "You are a support triage assistant.\n"
        "Classify the ticket and suggest a reply. Be brief.\n"
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
    hello_world: bool = typer.Option(
        False, "--hello-world", help="Skip dataset; just make one Respan-logged call."
    ),
) -> None:
    """Run a variant against the eval set via Respan Gateway."""
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

    engine = _engine(settings)
    prompt_path = Path("prompts") / f"{variant}.md"
    if not prompt_path.exists():
        console.print(f"[red]Missing prompt file: {prompt_path}[/red]")
        raise typer.Exit(code=1)
    if not dataset.exists():
        console.print(f"[red]Missing dataset: {dataset}[/red]")
        raise typer.Exit(code=1)

    prompt_text = prompt_path.read_text()
    with session(engine) as s:
        v = Variant(name=variant, prompt=prompt_text, model=settings.default_model)
        s.add(v)
        s.commit()
        s.refresh(v)
        r = Run(variant_id=v.id, dataset=str(dataset), status="running")
        s.add(r)
        s.commit()
        s.refresh(r)

    table = Table("case", "ok", "tokens", "latency", title=f"run {r.id} — {variant}")
    done = 0
    with span("arena.run", variant=variant, run_id=r.id), dataset.open() as fh:
        for line in fh:
            if done >= cases:
                break
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_inputs = case.get("inputs", {})
            user_text = case_inputs.get("ticket") or json.dumps(case_inputs)
            try:
                with span("arena.case", case_id=case.get("id", "")):
                    resp = client.chat(
                        [
                            {"role": "system", "content": prompt_text},
                            {"role": "user", "content": user_text},
                        ],
                        temperature=0.0,
                    )
                table.add_row(
                    case.get("id", "?"),
                    "[green]✓[/green]",
                    str(resp.total_tokens),
                    f"{resp.latency_ms}ms",
                )
            except Exception as exc:
                table.add_row(case.get("id", "?"), f"[red]{exc}[/red]", "-", "-")
            done += 1

    with session(engine) as s:
        updated = s.get(Run, r.id)
        if updated is not None:
            updated.status = "done"
            updated.completed_cases = done
            s.add(updated)
            s.commit()

    console.print(table)
    console.print(f"Run id: [bold]{r.id}[/bold]  |  cases: {done}")


# ---------------------------------------------------------------- placeholders

@app.command()
def judge(run_id: str = typer.Argument(...)) -> None:
    """Score a run with the configured judge ensemble. (Week-1 stub.)"""
    console.print(f"[yellow]TODO[/yellow] judge run {run_id} — Week 1 Day 5.")


@app.command()
def optimize(
    variant: str = typer.Argument(...),
    budget: int = typer.Option(30, "--budget"),
    target: str = typer.Option("reply_quality", "--target"),
) -> None:
    """Auto-propose prompt variants. (Week-2 stub.)"""
    console.print(
        f"[yellow]TODO[/yellow] optimize {variant} budget={budget} target={target}"
        " — Week 2 Day 8."
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
    console.print(f"[yellow]TODO[/yellow] gate vs baseline={baseline} — Week 2 Day 10.")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
