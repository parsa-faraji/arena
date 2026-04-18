"""Tracing SDK integration — both the present and the absent path.

The production SDK is published under two package names during the
keywords-ai -> respan rename, so the import resolver has two branches.
We exercise the present path with a lightweight stand-in so the
expectation ('init is called, span wraps the work') doesn't rot.
"""
from __future__ import annotations

import types
from typing import Any

import arena.tracing as tracing


class _SpanRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, *, name: str, **attrs: Any) -> _SpanCtx:
        self.calls.append((name, attrs))
        return _SpanCtx()


class _SpanCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> None:
        return None


def _fake_sdk(recorder: _SpanRecorder) -> types.SimpleNamespace:
    module = types.SimpleNamespace()
    module.init = lambda api_key, app_name: None  # type: ignore[attr-defined]
    module.workflow = recorder  # type: ignore[attr-defined]
    return module


def test_span_delegates_to_sdk_when_loaded(monkeypatch) -> None:
    recorder = _SpanRecorder()
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    monkeypatch.setattr(tracing, "_load_sdk", lambda: _fake_sdk(recorder))

    tracing.init_tracing(api_key="sk-test", app_name="arena-test")
    with tracing.span("arena.example", run_id="r1"):
        pass

    assert recorder.calls == [("arena.example", {"run_id": "r1"})]


def test_span_is_noop_without_api_key(monkeypatch) -> None:
    # If there's no API key we must NOT load the SDK, even if it's installed.
    called: list[bool] = []
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    monkeypatch.setattr(tracing, "_load_sdk", lambda: called.append(True) or _fake_sdk(_SpanRecorder()))

    tracing.init_tracing(api_key=None)
    with tracing.span("arena.example"):
        pass
    assert called == []


def test_span_survives_sdk_failures(monkeypatch) -> None:
    class _Blowup:
        def __call__(self, **_: Any) -> Any:
            raise RuntimeError("boom")

    bad_sdk = types.SimpleNamespace()
    bad_sdk.init = lambda api_key, app_name: None  # type: ignore[attr-defined]
    bad_sdk.workflow = _Blowup()  # type: ignore[attr-defined]
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    monkeypatch.setattr(tracing, "_load_sdk", lambda: bad_sdk)

    tracing.init_tracing(api_key="sk-test")
    # Must not raise — instrumentation bugs can't take down the caller.
    with tracing.span("arena.example"):
        pass
