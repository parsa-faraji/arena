"""End-to-end miner: trace source -> clusters -> labels -> eval cases."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arena.evals.dataset import EvalCase
from arena.mine.cluster import Cluster, cluster_traces
from arena.mine.label import label_cluster
from arena.mine.source import Trace, TraceSource

if TYPE_CHECKING:
    from arena.gateway.client import GatewayClient

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MiningReport:
    total_traces: int
    clusters: list[Cluster]
    cases: list[EvalCase]


def mine_to_eval_cases(
    source: TraceSource,
    *,
    client: GatewayClient | None = None,
    label_model: str = "gpt-4o-mini",
    min_cluster_size: int = 3,
    max_traces: int = 500,
    only_failures: bool = True,
    label_clusters: bool = True,
    max_cases_per_cluster: int | None = None,
) -> MiningReport:
    """Pull traces, cluster them, label, and emit `EvalCase`s.

    If `client` is provided and `label_clusters=True`, clusters get
    LLM-generated labels. Without a client we fall back to a cheap
    heuristic — this keeps the miner usable offline for demos and tests.
    """
    traces = list(source.pull(limit=max_traces, only_failures=only_failures))
    log.info("mined %d traces (only_failures=%s)", len(traces), only_failures)

    clusters = cluster_traces(traces, min_cluster_size=min_cluster_size)

    if label_clusters and client is not None:
        for cluster in clusters:
            cluster.label = label_cluster(cluster, client=client, model=label_model)
    else:
        from arena.mine.label import _heuristic_label

        for cluster in clusters:
            cluster.label = _heuristic_label(cluster)

    cases = _cases_from_clusters(clusters, max_per_cluster=max_cases_per_cluster)
    return MiningReport(total_traces=len(traces), clusters=clusters, cases=cases)


def _cases_from_clusters(
    clusters: Iterable[Cluster],
    *,
    max_per_cluster: int | None,
) -> list[EvalCase]:
    out: list[EvalCase] = []
    for cluster in clusters:
        members = (
            cluster.members[:max_per_cluster] if max_per_cluster else cluster.members
        )
        for trace in members:
            out.append(_case_from_trace(trace, cluster))
    return out


def _case_from_trace(trace: Trace, cluster: Cluster) -> EvalCase:
    tags = list(trace.tags)
    if cluster.label:
        tags.append(f"cluster:{cluster.label}")
    return EvalCase(
        id=trace.trace_id or f"trace-{id(trace)}",
        inputs={"ticket": trace.user_text},
        expected=None,  # human labels come later; the cluster label is a hint
        tags=tags,
        source="respan-trace",
        trace_id=trace.trace_id or None,
    )
