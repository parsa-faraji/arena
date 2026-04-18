"""End-to-end CLI drive against a fake Respan gateway.

Runs every public arena command in sequence. No real network calls — a
`FakeGateway` stands in for the real Respan Gateway, producing plausible
outputs for classification, judging, pairwise comparison, and cluster
labelling.

Useful for:
- Local smoke-testing the CLI after a refactor.
- Screencasting / demoing without needing Respan credentials.
- Capturing the "money shot" output for the README.

Run from this directory::

    python demo_drive.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("RESPAN_API_KEY", "sk-fake-demo-key")
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{Path('./arena.db').absolute()}"
)
os.environ.setdefault("ARENA_DISABLE_LOG", "true")

from arena import cli  # noqa: E402
from arena.gateway.client import GatewayResponse  # noqa: E402
from typer.testing import CliRunner  # noqa: E402


class FakeGateway:
    """Route-aware stand-in: triage for run, rubric score for judges,
    pairwise winner for head-to-head, topic label for the miner."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **_):
        self.calls += 1
        user = messages[-1]["content"].lower() if messages else ""
        system = next(
            (m["content"].lower() for m in messages if m["role"] == "system"),
            "",
        )

        if "strict but fair evaluator" in system:
            if "output 1" in user and "output 2" in user:
                # Pairwise — make A win 65% of the time for a legible demo.
                h = int(hashlib.sha1(user.encode()).hexdigest(), 16)
                winner = "1" if h % 4 != 0 else "2"
                rationale = "more actionable" if winner == "1" else "more concise"
                content = f'{{"winner": "{winner}", "rationale": "{rationale}"}}'
            else:
                content = '{"score": 4, "rationale": "clear and actionable"}'
        elif "name the common theme" in system:
            if any(k in user for k in ("refund", "charge", "invoice", "billing")):
                content = '{"label": "duplicate billing"}'
            elif any(k in user for k in ("login", "okta", "credentials", "sso")):
                content = '{"label": "login issues"}'
            elif any(k in user for k in ("webhook", "dark mode", "feature")):
                content = '{"label": "feature requests"}'
            elif any(k in user for k in ("crash", "500", "hang")):
                content = '{"label": "bugs"}'
            else:
                content = '{"label": "general"}'
        else:
            if any(k in user for k in ("refund", "charge", "billing", "invoice")):
                urgency, category = "high", "billing"
            elif any(k in user for k in ("login", "credentials", "okta", "sso")):
                urgency, category = "high", "account"
            elif any(k in user for k in ("500", "crash", "hang", "broken")):
                urgency, category = "high", "bug"
            elif any(k in user for k in ("feature", "webhook", "dark mode")):
                urgency, category = "low", "feature"
            else:
                urgency, category = "low", "other"
            content = (
                f'{{"urgency": "{urgency}", "category": "{category}", '
                f'"suggested_reply": "Got it — I\'ll take a look."}}'
            )

        return GatewayResponse(
            content=content, model="gpt-4o-mini",
            input_tokens=40, output_tokens=20, latency_ms=5, raw={},
        )


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_and_show(args: list[str]) -> str:
    fake = FakeGateway()
    runner = CliRunner()
    with (
        patch.object(cli, "GatewayClient", lambda **_: fake),
        patch.object(cli, "init_tracing", lambda **_: None),
    ):
        result = runner.invoke(cli.app, args)
    print(result.stdout)
    if result.exit_code != 0:
        print(f"[exit={result.exit_code}]")
    return result.stdout


def extract_run_id(stdout: str) -> str | None:
    match = re.search(r"Run id:\s*\x1b\[1m([a-f0-9]+)\x1b\[0m", stdout)
    if match:
        return match.group(1)
    match = re.search(r"Run id:\s+([a-f0-9]+)", stdout)
    return match.group(1) if match else None


def main() -> int:
    # Drive from the directory containing arena.config.yaml + prompts/.
    here = Path(__file__).parent
    os.chdir(here)

    banner("1. arena mine --fixture  (pull & cluster traces)")
    run_and_show([
        "mine", "--fixture", "fixtures/traces.jsonl",
        "--min-cluster", "3", "--all",
        "--output", "mined.jsonl",
    ])

    banner("2. arena run v0-baseline  (20 cases from dataset.jsonl)")
    out_a = run_and_show(["run", "v0-baseline", "--cases", "20", "--concurrency", "4"])
    run_a = extract_run_id(out_a)

    banner("3. arena run v1-optimized  (20 cases)")
    out_b = run_and_show(["run", "v1-optimized", "--cases", "20"])
    run_b = extract_run_id(out_b)

    banner("4. arena runs  (list the two runs)")
    run_and_show(["runs", "--limit", "5"])

    if run_a:
        banner(f"5. arena show {run_a[:8]}  (per-case detail)")
        run_and_show(["show", run_a[:8]])

    if run_a and run_b:
        banner(f"6. arena judge {run_a[:8]} --pairwise --vs {run_b[:8]}")
        run_and_show(["judge", run_a[:8], "--pairwise", "--vs", run_b[:8]])

    banner("done")
    print(f"run_a (v0-baseline): {run_a}")
    print(f"run_b (v1-optimized): {run_b}")
    print(f"database: {os.environ['DATABASE_URL']}")
    print("mined.jsonl: 35 traces -> 4 clusters -> 35 eval cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
