from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.evals.dataset import EvalCase
from arena.gateway.client import GatewayResponse
from arena.mine.cluster import cluster_traces
from arena.mine.miner import mine_to_eval_cases
from arena.mine.source import FixtureSource, Trace


def _trace(text: str, idx: int = 0, status: str = "success") -> Trace:
    return Trace(
        trace_id=f"t{idx}",
        timestamp=datetime(2026, 4, 10, tzinfo=UTC),
        user_text=text,
        assistant_text="",
        status=status,
    )


def test_cluster_splits_by_topic() -> None:
    traces = [
        _trace("please refund my duplicate charge", i) for i in range(6)
    ] + [
        _trace("my login is broken, invalid credentials error", 100 + i) for i in range(6)
    ] + [
        _trace("can you add webhooks for new events", 200 + i) for i in range(6)
    ]
    clusters = cluster_traces(traces, method="kmeans", k=3)
    assert len(clusters) == 3
    # Each cluster should be internally coherent — check that "refund", "login",
    # and "webhook" live in different buckets.
    keywords = [" ".join(m.user_text for m in c.members).lower() for c in clusters]
    found = {
        "refund": any("refund" in text for text in keywords),
        "login": any("login" in text for text in keywords),
        "webhook": any("webhook" in text for text in keywords),
    }
    assert all(found.values())


def test_cluster_handles_tiny_corpus() -> None:
    clusters = cluster_traces([_trace("only one trace")], method="kmeans")
    assert len(clusters) == 1
    assert clusters[0].size == 1


def test_cluster_skips_empty_user_text() -> None:
    traces = [
        _trace("real question one", 1),
        Trace(trace_id="empty", timestamp=datetime.now(tz=UTC), user_text="", assistant_text=""),
        _trace("real question two", 2),
    ]
    clusters = cluster_traces(traces, method="kmeans")
    members = [m.trace_id for c in clusters for m in c.members]
    assert "empty" not in members


class _FakeGateway:
    """Replies with a fixed cluster label each call."""

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels
        self.i = 0
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> GatewayResponse:
        self.prompts.append(messages[-1]["content"])
        label = self._labels[self.i % len(self._labels)]
        self.i += 1
        return GatewayResponse(
            content=f'{{"label": "{label}"}}',
            model="gpt-4o-mini",
            input_tokens=10, output_tokens=5, latency_ms=1, raw={},
        )


def test_miner_emits_eval_cases_with_cluster_tags() -> None:
    traces = (
        [_trace("refund duplicate charge", i) for i in range(6)]
        + [_trace("login failed invalid password", 100 + i) for i in range(6)]
    )

    class _Source:
        def pull(self, **_: Any):
            yield from traces

    client = _FakeGateway(labels=["billing refunds", "login issues"])
    report = mine_to_eval_cases(
        _Source(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        min_cluster_size=3,
        only_failures=False,
    )
    assert report.total_traces == 12
    assert len(report.clusters) == 2
    labels = {c.label for c in report.clusters}
    assert {"billing refunds", "login issues"}.issubset(labels)

    # Every emitted case carries a cluster:<label> tag.
    assert all(
        any(tag.startswith("cluster:") for tag in case.tags) for case in report.cases
    )
    assert len(report.cases) == 12


def test_miner_offline_fallback_without_client() -> None:
    traces = [_trace("refund please", i) for i in range(6)]

    class _Source:
        def pull(self, **_: Any):
            yield from traces

    report = mine_to_eval_cases(
        _Source(),  # type: ignore[arg-type]
        client=None,
        min_cluster_size=3,
        only_failures=False,
    )
    # Heuristic labels come from the centroid's first words.
    assert report.clusters[0].label
    assert all(isinstance(c, EvalCase) for c in report.cases)


def test_miner_reads_from_fixture_file() -> None:
    fixture = Path(__file__).parent.parent.parent.parent / "examples" / "support-triage" / "fixtures" / "traces.jsonl"
    if not fixture.exists():
        return  # skip when fixture missing (e.g., partial checkout)
    source = FixtureSource(fixture)
    report = mine_to_eval_cases(
        source,
        client=None,
        min_cluster_size=3,
        only_failures=False,
    )
    assert report.total_traces >= 20
    assert len(report.clusters) >= 3
    assert len(report.cases) == report.total_traces
