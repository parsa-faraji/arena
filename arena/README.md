# arena (Python package)

The Python core of Arena. Install with `pip install -e .` from this directory.

See the [top-level README](../README.md) for the project pitch.

## Layout

```
src/arena/
├── cli.py          # typer entrypoint — `arena` command
├── config.py       # pydantic settings (env + .env loading)
├── tracing.py      # respan SDK wiring
├── gate.py         # CI regression gate
├── gateway/        # Respan Gateway client, semantic cache, fallback chain
├── evals/          # dataset, runner, evaluators
├── judges/         # pairwise, rubric, reference, ensemble
├── optimizer/      # ProTeGi textual gradients, bootstrapped few-shot
├── mine/           # pull + cluster + label Respan traces
└── store/          # sqlmodel tables + migrations
```

## Development

```bash
pip install -e '.[dev,mine,tracing]'
pytest
ruff check src tests
mypy src
```
