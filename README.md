# Arena

**An LLM regression-testing gym for [Respan](https://respan.ai). Jest for prompts, Optuna for prompt tuning — wired into your Respan workspace.**

Arena closes the loop between Respan's production observability and your prompt iteration cycle:

1. **Mine** — pull failing production traces from your Respan workspace and cluster them into labelled failure modes.
2. **Run** — replay hand-written + auto-mined eval cases against any prompt/model variant, routed through Respan Gateway with a semantic cache and fallback chain.
3. **Judge** — score each run with pluggable LLM-as-judge pipelines (pairwise, rubric, reference-based, ensemble).
4. **Optimize** — auto-propose prompt variants with a textual-gradient optimizer (ProTeGi + bootstrapped few-shot) under a configurable budget.
5. **Gate** — block PRs that regress quality with a GitHub Actions check, complete with confidence intervals.

Every LLM call is logged to your Respan workspace, so the entire experiment graph (variants → judges → optimizer steps) shows up as a single trace tree in the Respan UI.

## Why this exists

Respan's pitch is *proactive* observability — catch problems before they hit production. Arena is the opinionated "after you've caught it, now what?" layer: a mechanical loop from *failure observed in Respan* → *regression test* → *optimized prompt* → *PR gate that prevents recurrence*. The loop is not magic; it's just four well-tested pieces glued together cleanly.

## Quick start

```bash
# 1. Install (requires Python 3.11+)
pip install -e ./arena

# 2. Scaffold a new project (creates arena.config.yaml, sample dataset)
arena init my-project && cd my-project

# 3. Drop in your Respan credentials
cp .env.example .env && $EDITOR .env

# 4. Run the baseline variant on 20 cases
arena run v0 --cases 20

# 5. Score it with the default judge ensemble
arena judge <run-id>

# 6. Try to beat it
arena optimize v0 --budget 30 --target reply_quality

# 7. Use the CI gate in a GitHub Action
arena gate --baseline <baseline-run-id>
```

## Demo: Support ticket triage

The `examples/support-triage/` project ships a deliberately mediocre `v0` prompt and a 200-case eval set. Arena's optimizer reliably beats `v0` by 5+ points on `reply_quality` within a 30-call budget — the full demo runs in under 3 minutes.

## Architecture

```
  Respan traces ─┐
                 ├─> eval set
  hand-written ──┘        │
                          v
  Variant ──> Respan Gateway ──> LLM-as-judge
                                     │
   optimizer <── judge scores ───────┤
                                     │
                                     v
                       Dashboard (arena-web) + CI gate
```

See `docs/architecture.md` for the long version.

## Repo layout

- `arena/` — Python package (CLI + library)
- `arena-web/` — Next.js dashboard (read-only viewer)
- `examples/support-triage/` — end-to-end demo project
- `.github/workflows/` — CI + regression gate + Respan smoke test

## License

MIT — see [LICENSE](./LICENSE).
