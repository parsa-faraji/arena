"""End-to-end CLI drive against a fake Respan gateway.

Runs every public arena command in sequence. No real network calls — a
`FakeGateway` stands in for the real Respan Gateway, producing plausible
outputs for classification, judging, pairwise comparison, cluster
labelling, and prompt-optimizer proposals.

The fake is **variant-aware**: it reads the triage system prompt and
produces terser outputs for the v0 baseline than for the v1 optimized
prompt (which advertises a length + tone contract). Rubric judges, in
turn, reward the richer v1 outputs — so the demo tells an honest
"optimizer beats baseline" story without needing a live LLM.

Useful for:
- Local smoke-testing the CLI after a refactor.
- Screencasting / demoing without needing Respan credentials.
- Seeding the Next.js dashboard with a believable run history.

Run from this directory::

    python demo_drive.py
"""
from __future__ import annotations

import hashlib
import json
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


# Canned prompt the optimizer "discovers" when shown the v0 baseline.
# Mirrors the substance of prompts/v1-optimized.md — length contract,
# urgency rubric, tone guidance, few-shot examples. This is the payload
# returned by FakeGateway when the optimizer asks for a proposal.
_OPTIMIZED_PROPOSAL = """\
You are a support triage assistant for a B2B SaaS product. Downstream
automation parses your JSON, so correctness of the JSON matters more
than anything else.

Given an inbound ticket, return a JSON object with:
- urgency: "low" | "med" | "high"
  - "high" = user blocked, revenue at risk, or data loss
  - "med" = clear problem but user has a workaround
  - "low" = question, feedback, or nice-to-have
- category: "billing" | "bug" | "feature" | "account" | "other"
- suggested_reply: 1–3 sentences. Acknowledge the specific issue, set
  expectations with a concrete next step, and sound like a real human
  (avoid "sorry for the inconvenience").

Examples:

Ticket: "I was charged twice for my subscription this month."
Output: {"urgency": "high", "category": "billing", "suggested_reply": "That's on us — I can see the duplicate charge and I'll refund it today. You'll get an email confirmation within a couple hours."}

Ticket: "It would be nice if you supported dark mode."
Output: {"urgency": "low", "category": "feature", "suggested_reply": "Appreciate the suggestion — I've added it to our backlog. No ETA yet, but it's on the radar."}

Return ONLY the JSON object, no prose.
"""


_V1_MARKERS = ("1–3 sentences", "set expectations", "avoid")


def _is_v1_prompt(system_prompt: str) -> bool:
    return any(marker in system_prompt for marker in _V1_MARKERS)


def _pick_urgency_category(user: str) -> tuple[str, str]:
    if any(k in user for k in ("refund", "charge", "billing", "invoice")):
        return "high", "billing"
    if any(k in user for k in ("login", "credentials", "okta", "sso")):
        return "high", "account"
    if any(k in user for k in ("500", "crash", "hang", "broken")):
        return "high", "bug"
    if any(k in user for k in ("feature", "webhook", "dark mode")):
        return "low", "feature"
    return "low", "other"


def _v0_reply(user: str, category: str) -> str:
    # Terse, generic — what a thin prompt produces.
    if category == "billing":
        return "Got it — I'll take a look."
    if category == "account":
        return "We'll look into your login issue."
    if category == "bug":
        return "Thanks for flagging."
    if category == "feature":
        return "Thanks for the suggestion."
    return "Got it — I'll take a look."


def _v1_reply(user: str, category: str) -> str:
    # Longer, action-bearing — matches the v1 contract.
    if category == "billing":
        return (
            "Thanks for flagging — I can see the duplicate billing on your account. "
            "I've already queued the refund, and you'll see confirmation in your "
            "inbox within a few hours."
        )
    if category == "account":
        return (
            "That's frustrating — I'll unblock your sign-in right now. "
            "You should be able to log in within the next 10 minutes; "
            "reply here if anything's still off."
        )
    if category == "bug":
        return (
            "Appreciate the detailed report — I'm opening a bug and pinging the "
            "team. I'll follow up with an ETA once we've reproduced it."
        )
    if category == "feature":
        return (
            "Good idea — I've logged it for the roadmap review. "
            "I'll ping you if it ships so you're the first to hear."
        )
    return "Thanks — I'll take a look and get back with a concrete next step."


def _extract_output(user_block: str, marker: str) -> str:
    """Pull the raw text that followed ``Output 1:`` / ``Output 2:`` markers
    in the pairwise judge prompt."""
    idx = user_block.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    # Pairwise prompt separates blocks with a blank line.
    end = user_block.find("\n\n", start)
    return user_block[start:end if end != -1 else None].strip()


def _rubric_score(user_block: str) -> int:
    """Score the output embedded in a rubric judge's prompt on 1..5.

    Looks for the "Model output to judge:" block and ranks it on length +
    presence of action words. Rich v1 outputs land at 4–5; terse v0 at 2–3.
    """
    match = re.search(
        r"Model output to judge:\s*(.+?)(?:\n\nReturn JSON|$)",
        user_block,
        re.DOTALL,
    )
    output = match.group(1).strip() if match else user_block
    length = len(output)
    action_phrases = (
        "follow up",
        "next step",
        "queued",
        "within",
        "unblock",
        "confirmation",
        "logged it",
        "concrete",
        "reply here",
        "we've",
    )
    hits = sum(1 for p in action_phrases if p in output.lower())
    if length > 180 or hits >= 2:
        return 5
    if length > 120 or hits >= 1:
        return 4
    if length > 60:
        return 3
    return 2


class FakeGateway:
    """Route-aware stand-in: triage for run, rubric score for judges,
    pairwise winner for head-to-head, topic label for the miner, prompt
    proposal for the optimizer."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **_):
        self.calls += 1
        user = messages[-1]["content"] if messages else ""
        user_lower = user.lower()
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "",
        )
        system_lower = system.lower()

        # --- prompt optimizer ----------------------------------------------
        if "prompt engineer" in system_lower:
            # On the first proposal we hand back the v1-optimized prompt;
            # subsequent proposals return minor paraphrases so the greedy
            # climb halts naturally after one promotion.
            if "1–3 sentences" in user or "set expectations" in user:
                content = json.dumps(
                    {
                        "prompt": _OPTIMIZED_PROPOSAL.rstrip()
                        + "\n\nBe decisive — the customer wants a resolution, not an apology.",
                        "gradient": "emphasize decisiveness and avoid hedging language",
                    }
                )
            else:
                content = json.dumps(
                    {
                        "prompt": _OPTIMIZED_PROPOSAL,
                        "gradient": (
                            "add length + tone contract, urgency rubric, few-shot "
                            "examples so the reply is specific and decisive"
                        ),
                    }
                )
        # --- judge (rubric / pairwise) -------------------------------------
        elif "strict but fair evaluator" in system_lower:
            if "output 1" in user_lower and "output 2" in user_lower:
                # Pairwise — prefer the longer/richer reply 80% of the time,
                # with small ties and noise to keep it realistic.
                one = _extract_output(user, "Output 1:")
                two = _extract_output(user, "Output 2:")
                h = int(hashlib.sha1(user.encode()).hexdigest(), 16)
                noise = h % 10
                if abs(len(one) - len(two)) < 15:
                    winner = "tie"
                    rationale = "roughly equivalent"
                elif len(one) > len(two):
                    winner = "1" if noise < 8 else ("2" if noise < 9 else "tie")
                    rationale = "more specific next step"
                else:
                    winner = "2" if noise < 8 else ("1" if noise < 9 else "tie")
                    rationale = "more specific next step"
                content = json.dumps({"winner": winner, "rationale": rationale})
            else:
                score = _rubric_score(user)
                content = json.dumps(
                    {
                        "score": score,
                        "rationale": (
                            "clear next step and tone"
                            if score >= 4
                            else "acknowledges issue but vague on next step"
                        ),
                    }
                )
        # --- cluster labeller ----------------------------------------------
        elif "name the common theme" in system_lower:
            if any(k in user_lower for k in ("refund", "charge", "invoice", "billing")):
                label = "duplicate billing"
            elif any(k in user_lower for k in ("login", "okta", "credentials", "sso")):
                label = "login issues"
            elif any(k in user_lower for k in ("webhook", "dark mode", "feature")):
                label = "feature requests"
            elif any(k in user_lower for k in ("crash", "500", "hang")):
                label = "bugs"
            else:
                label = "general"
            content = json.dumps({"label": label})
        # --- triage (the actual variant under test) ------------------------
        else:
            urgency, category = _pick_urgency_category(user_lower)
            reply = (
                _v1_reply(user_lower, category)
                if _is_v1_prompt(system)
                else _v0_reply(user_lower, category)
            )
            content = json.dumps(
                {
                    "urgency": urgency,
                    "category": category,
                    "suggested_reply": reply,
                }
            )

        return GatewayResponse(
            content=content,
            model="gpt-4o-mini",
            input_tokens=40,
            output_tokens=20 if not _is_v1_prompt(system) else 55,
            latency_ms=5,
            raw={},
        )


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_and_show(args: list[str], *, allow_fail: bool = False) -> tuple[str, int]:
    fake = FakeGateway()
    runner = CliRunner()
    with (
        patch.object(cli, "GatewayClient", lambda **_: fake),
        patch.object(cli, "init_tracing", lambda **_: None),
    ):
        result = runner.invoke(cli.app, args)
    print(result.stdout)
    if result.exit_code != 0 and not allow_fail:
        print(f"[exit={result.exit_code}]")
    return result.stdout, result.exit_code


def extract_run_id(stdout: str) -> str | None:
    match = re.search(r"Run id:\s*\x1b\[1m([a-f0-9]+)\x1b\[0m", stdout)
    if match:
        return match.group(1)
    match = re.search(r"Run id:\s+([a-f0-9]+)", stdout)
    return match.group(1) if match else None


def extract_best_run_id(stdout: str) -> str | None:
    match = re.search(r"best run id\s+([a-f0-9]{16})", stdout)
    return match.group(1) if match else None


def main() -> int:
    here = Path(__file__).parent
    os.chdir(here)

    banner("1. arena mine --fixture  (pull & cluster traces)")
    run_and_show([
        "mine", "--fixture", "fixtures/traces.jsonl",
        "--min-cluster", "3", "--all",
        "--output", "mined.jsonl",
    ])

    banner("2. arena run v0-baseline  (20 cases from dataset.jsonl)")
    out_a, _ = run_and_show(["run", "v0-baseline", "--cases", "20", "--concurrency", "4"])
    run_a = extract_run_id(out_a)

    banner("3. arena run v1-optimized  (20 cases, richer prompt)")
    out_b, _ = run_and_show(["run", "v1-optimized", "--cases", "20"])
    run_b = extract_run_id(out_b)

    banner("4. arena runs  (list the runs)")
    run_and_show(["runs", "--limit", "10"])

    if run_a:
        banner(f"5. arena show {run_a[:8]}  (per-case detail)")
        run_and_show(["show", run_a[:8]])

    if run_a and run_b:
        banner(f"6. arena judge {run_a[:8]} --pairwise --vs {run_b[:8]}")
        run_and_show(["judge", run_a[:8], "--pairwise", "--vs", run_b[:8]])

    banner("7. arena optimize v0-baseline --budget 3 --target reply_quality")
    out_opt, _ = run_and_show([
        "optimize", "v0-baseline",
        "--budget", "3",
        "--target", "reply_quality",
        "--cases", "20",
    ])
    best_run_id = extract_best_run_id(out_opt)

    if best_run_id and run_a:
        banner(
            f"8. arena gate --baseline {run_a[:8]} --run {best_run_id[:8]}  (expect PASS)"
        )
        run_and_show([
            "gate",
            "--baseline", run_a[:8],
            "--run", best_run_id[:8],
        ])

        banner(
            f"9. arena gate --baseline {run_b[:8]} --run {run_a[:8]}  (expect FAIL)"
        )
        run_and_show(
            [
                "gate",
                "--baseline", run_b[:8],
                "--run", run_a[:8],
            ],
            allow_fail=True,
        )

    banner("done")
    print(f"run_a (v0-baseline): {run_a}")
    print(f"run_b (v1-optimized): {run_b}")
    print(f"optimizer best run: {best_run_id}")
    print(f"database: {os.environ['DATABASE_URL']}")
    print("mined.jsonl: 35 traces -> 4 clusters -> 35 eval cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
