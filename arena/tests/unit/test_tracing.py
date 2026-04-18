from __future__ import annotations

import arena.tracing as tracing


def test_span_noop_without_sdk(monkeypatch) -> None:
    # Force a fresh state; no SDK installed in CI by default.
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    tracing.init_tracing(api_key=None)
    with tracing.span("example", foo="bar"):
        pass  # must not raise


def test_init_twice_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    tracing.init_tracing(api_key=None)
    tracing.init_tracing(api_key="sk-test")
    assert tracing._init_attempted is True
