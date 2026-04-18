# Support triage demo

The headline example for Arena. A deliberately mediocre `v0-baseline.md`
prompt classifies support tickets and drafts replies; Arena's optimizer
should reliably beat it on `reply_quality` within a 30-call budget.

## Run it against a real Respan workspace

```bash
# from repo root, with .venv active and RESPAN_API_KEY set:
cd examples/support-triage

arena mine --from-respan --last 24h       # pull + cluster real traces
arena run v0-baseline --cases 20          # baseline run
arena run v1-optimized --cases 20         # challenger
arena runs                                # list runs
arena show <run-id>                       # per-case detail
arena judge <run-a> --pairwise --vs <run-b>  # head-to-head
```

## Run the offline demo (no credentials needed)

The demo driver exercises every CLI command against a fake gateway, so
you can screencast the whole flow without hitting Respan:

```bash
python demo_drive.py
```

Expected output includes: 35 traces mined into 4 clusters, two runs
scored across 5 evaluators (exact-match + json-parse + rubric +
ensemble), a pairwise shoot-out with a ~65% win rate for the challenger.

## What's in here

- `dataset.jsonl` — 20 seed cases with ground-truth `urgency` + `category`
  (the LLM-as-judge handles `suggested_reply`). Mine more from your Respan
  workspace with `arena mine --from-respan`.
- `prompts/v0-baseline.md` — the deliberately mediocre starting prompt.
- `prompts/v1-optimized.md` — a hand-tuned reference for the optimizer to
  target; delete before running the optimizer to see it rediscover gains
  from scratch.
- `arena.config.yaml` — judges, evaluators, and the fallback chain.
- `fixtures/traces.jsonl` — 35 fake Respan traces used by the offline demo.
- `demo_drive.py` — scripted end-to-end drive of every CLI command.
