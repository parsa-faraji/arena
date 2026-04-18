# Support triage demo

The headline example for Arena. A deliberately mediocre `v0-baseline.md`
prompt classifies support tickets and drafts replies; Arena's optimizer
should reliably beat it on `reply_quality` within a 30-call budget.

## Run it

```bash
# from repo root, with .venv active and RESPAN_API_KEY set:
cd examples/support-triage

arena run v0-baseline --cases 20         # baseline run
arena judge <run-id>                      # score it
arena optimize v0-baseline --budget 30    # try to beat it
arena gate --baseline <baseline-run-id>   # CI gate
```

## What's in here

- `dataset.jsonl` — 20 seed cases with ground-truth `urgency` + `category`
  (the LLM-as-judge handles `suggested_reply`). Mine more from your Respan
  workspace with `arena mine --from-respan`.
- `prompts/v0-baseline.md` — the deliberately mediocre starting prompt.
- `prompts/v1-optimized.md` — a hand-tuned reference for the optimizer to
  target; delete before running the optimizer to see it rediscover gains
  from scratch.
- `arena.config.yaml` — judges, evaluators, and the fallback chain.
