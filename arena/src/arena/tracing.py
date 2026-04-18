"""Respan tracing SDK wiring.

The Respan Python SDK is published under both `respan` (new name) and
`keywords_ai_tracing` (old name). We try both and fall back to a no-op
tracer so Arena still runs in environments without the SDK installed.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_tracer: Any = None
_init_attempted = False


def _load_sdk() -> Any | None:
    """Import the Respan tracing SDK if available. Prefer the new package name."""
    try:
        import respan as sdk  # type: ignore

        return sdk
    except ImportError:
        pass
    try:
        import keywordsai_tracing as sdk  # type: ignore

        return sdk
    except ImportError:
        return None


def init_tracing(*, api_key: str | None, app_name: str = "arena") -> None:
    """Initialise Respan tracing. Safe to call multiple times."""
    global _tracer, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    if not api_key:
        log.info("tracing: no RESPAN_API_KEY, running without Respan tracing")
        return

    sdk = _load_sdk()
    if sdk is None:
        log.info("tracing: respan SDK not installed (pip install -e '.[tracing]')")
        return

    try:
        if hasattr(sdk, "init"):
            sdk.init(api_key=api_key, app_name=app_name)
        _tracer = sdk
        log.info("tracing: respan SDK initialised for app=%s", app_name)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("tracing: failed to initialise respan SDK: %s", exc)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Open a named span if the tracing SDK is available, otherwise a no-op."""
    if _tracer is None:
        yield
        return

    decorator = None
    for attr in ("workflow", "task", "span"):
        candidate = getattr(_tracer, attr, None)
        if callable(candidate):
            decorator = candidate
            break

    if decorator is None:
        yield
        return

    try:
        with decorator(name=name, **attributes):  # type: ignore[misc]
            yield
    except TypeError:
        try:
            with decorator(name):  # type: ignore[misc]
                yield
        except Exception:  # pragma: no cover
            yield
    except Exception:  # pragma: no cover
        yield
