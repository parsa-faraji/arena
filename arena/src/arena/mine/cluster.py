"""Cluster traces by topic so humans label N clusters instead of N traces.

We default to TF-IDF + KMeans because those ship with scikit-learn, which
is already a core dep. The `[mine]` extras unlock
sentence-transformers + HDBSCAN for better semantic clustering on
variable-size groups.

Cluster size heuristics:
- KMeans: k = clamp(sqrt(n/2), 3, 12). Good enough for a demo;
  users can override.
- HDBSCAN: `min_cluster_size` is the only knob; it auto-picks k.

Tiny corpora (< 6 traces) are all put in a single cluster — clustering
noise at that size is pointless.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from arena.mine.source import Trace

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Cluster:
    """One group of semantically-similar traces."""

    id: int
    members: list[Trace] = field(default_factory=list)
    label: str = ""
    centroid_text: str = ""  # representative member's user_text

    @property
    def size(self) -> int:
        return len(self.members)


def cluster_traces(
    traces: Iterable[Trace],
    *,
    method: str = "auto",
    k: int | None = None,
    min_cluster_size: int = 3,
    vectorizer: Any | None = None,
) -> list[Cluster]:
    """Group traces by the semantic similarity of their user messages.

    `method` ∈ {"auto", "kmeans", "hdbscan"}.  "auto" picks HDBSCAN if
    available, KMeans otherwise. Tests should pass `method="kmeans"`
    plus a deterministic `k` to keep outputs stable.
    """
    traces_list = [t for t in traces if t.user_text.strip()]
    n = len(traces_list)
    if n == 0:
        return []
    if n < max(min_cluster_size * 2, 6):
        # Not enough data to meaningfully cluster — one bucket.
        return [
            Cluster(
                id=0,
                members=traces_list,
                centroid_text=traces_list[0].user_text,
            )
        ]

    vec = vectorizer or _default_vectorizer()
    texts = [t.user_text for t in traces_list]
    matrix = vec.fit_transform(texts)

    labels = _dispatch(method, matrix, k=k, min_cluster_size=min_cluster_size)

    clusters: dict[int, Cluster] = {}
    for label_id, trace in zip(labels, traces_list, strict=True):
        bucket = clusters.setdefault(int(label_id), Cluster(id=int(label_id)))
        bucket.members.append(trace)

    # Pick a representative centroid per cluster as the member whose vector
    # is closest to the cluster mean — the most "typical" failure.
    for bucket in clusters.values():
        bucket.centroid_text = _pick_centroid(bucket.members, vec)

    # Sort clusters by size desc; noise (-1 from HDBSCAN) goes last.
    ordered = sorted(
        clusters.values(),
        key=lambda c: (0 if c.id != -1 else 1, -c.size),
    )
    # Re-number so downstream doesn't see negative ids.
    for i, bucket in enumerate(ordered):
        bucket.id = i
    return ordered


# ------------------------------------------------------------- internals

def _default_vectorizer() -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=2000,
        min_df=1,
    )


def _dispatch(method: str, matrix: Any, *, k: int | None, min_cluster_size: int) -> Any:
    if method == "auto":
        method = "hdbscan" if _hdbscan_available() else "kmeans"
    if method == "hdbscan":
        try:
            return _hdbscan(matrix, min_cluster_size=min_cluster_size)
        except Exception as exc:
            log.info("hdbscan unavailable (%s); falling back to kmeans", exc)
    return _kmeans(matrix, k=k)


def _kmeans(matrix: Any, *, k: int | None) -> Any:
    from sklearn.cluster import KMeans

    n = matrix.shape[0]
    if k is None:
        k = max(3, min(12, int(math.sqrt(n / 2))))
    k = min(k, n)
    model = KMeans(n_clusters=k, random_state=0, n_init="auto")
    return model.fit_predict(matrix)


def _hdbscan(matrix: Any, *, min_cluster_size: int) -> Any:
    import hdbscan  # type: ignore
    import numpy as np

    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    dense = np.asarray(dense, dtype="float64")
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    return model.fit_predict(dense)


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _pick_centroid(members: list[Trace], vec: Any) -> str:
    if not members:
        return ""
    if len(members) == 1:
        return members[0].user_text
    import numpy as np

    matrix = vec.transform([m.user_text for m in members])
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    mean = dense.mean(axis=0)
    distances = np.linalg.norm(dense - mean, axis=1)
    return members[int(np.argmin(distances))].user_text
