"""Label clusters with a short LLM-generated name.

Given a cluster of similar traces, we ask an LLM to name the failure
mode or topic in 2-5 words. That label becomes the `tag` on emitted
eval cases, so a reviewer reading `arena runs` can immediately see
whether a regression is concentrated in, say, "duplicate billing"
cases.

Labels stay intentionally short — long LLM-generated titles stop being
scannable.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arena.evals.evaluators import parse_json_output

if TYPE_CHECKING:
    from arena.gateway.client import GatewayClient
    from arena.mine.cluster import Cluster

log = logging.getLogger(__name__)

_LABEL_SYSTEM = (
    "You name the common theme in a set of customer messages. "
    "Reply ONLY with JSON."
)


def label_cluster(
    cluster: Cluster,
    *,
    client: GatewayClient,
    model: str = "gpt-4o-mini",
    sample_size: int = 8,
) -> str:
    """Return a short human-readable label for a cluster.

    Uses up to `sample_size` representative messages in the prompt so
    token cost stays bounded even on big clusters.
    """
    if cluster.size == 0:
        return ""
    sample = [m.user_text for m in cluster.members[:sample_size]]
    prompt = (
        "Here are customer messages that were grouped together:\n\n"
        + "\n---\n".join(sample)
        + "\n\n"
        'Return JSON: {"label": "2-5 word topic or failure mode"}.'
    )
    try:
        resp = client.chat(
            [
                {"role": "system", "content": _LABEL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        log.warning("label_cluster: gateway failed (%s); using fallback", exc)
        return _heuristic_label(cluster)

    parsed = parse_json_output(resp.content) or {}
    label = str(parsed.get("label", "")).strip()
    return label or _heuristic_label(cluster)


def _heuristic_label(cluster: Cluster) -> str:
    """Fallback label when the LLM call fails — use the centroid's first words."""
    words = cluster.centroid_text.split()[:4]
    return " ".join(words).strip().lower() or f"cluster-{cluster.id}"
