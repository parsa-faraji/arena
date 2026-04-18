from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from arena.mine.source import (
    FixtureSource,
    RespanAPI,
    Trace,
    _trace_from_row,
    parse_relative_duration,
)


def _write_fixture(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_fixture_source_yields_traces(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "traces.jsonl",
        [
            {
                "id": "a",
                "created_at": "2026-04-10T09:00:00Z",
                "prompt_messages": [{"role": "user", "content": "refund"}],
                "completion": "{}",
                "status": "success",
            },
            {
                "id": "b",
                "created_at": "2026-04-10T09:01:00Z",
                "prompt_messages": [{"role": "user", "content": "login"}],
                "completion": "{}",
                "status": "error",
            },
        ],
    )
    traces = list(FixtureSource(path).pull())
    assert [t.trace_id for t in traces] == ["a", "b"]
    assert traces[1].status == "error"
    assert traces[1].looks_like_failure is True


def test_fixture_source_filters_by_window(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "traces.jsonl",
        [
            {"id": "old", "created_at": "2026-04-01T00:00:00Z", "prompt_messages": [{"role": "user", "content": "x"}]},
            {"id": "new", "created_at": "2026-04-10T00:00:00Z", "prompt_messages": [{"role": "user", "content": "y"}]},
        ],
    )
    since = datetime(2026, 4, 5, tzinfo=UTC)
    traces = list(FixtureSource(path).pull(since=since))
    assert [t.trace_id for t in traces] == ["new"]


def test_fixture_source_only_failures(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "traces.jsonl",
        [
            {"id": "ok", "prompt_messages": [{"role": "user", "content": "x"}], "status": "success"},
            {"id": "bad", "prompt_messages": [{"role": "user", "content": "y"}], "status": "error"},
            {"id": "tagged", "prompt_messages": [{"role": "user", "content": "z"}], "tags": ["failure"]},
        ],
    )
    traces = list(FixtureSource(path).pull(only_failures=True))
    assert {t.trace_id for t in traces} == {"bad", "tagged"}


def test_fixture_source_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FixtureSource(tmp_path / "missing.jsonl")


def test_trace_from_row_handles_multiple_shapes() -> None:
    row = {
        "trace_id": "t1",
        "time": 1_712_745_600,
        "messages": [{"role": "user", "content": "hello"}],
        "completion_message": {"role": "assistant", "content": "hi there"},
        "model": "gpt-4o-mini",
        "latency": 123,
    }
    trace = _trace_from_row(row)
    assert trace.trace_id == "t1"
    assert trace.user_text == "hello"
    assert trace.model == "gpt-4o-mini"
    assert trace.latency_ms == 123


def test_respan_api_hits_configured_path() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "abc",
                        "created_at": "2026-04-10T00:00:00Z",
                        "prompt_messages": [{"role": "user", "content": "refund"}],
                        "completion": "ok",
                    }
                ]
            },
        )

    api = RespanAPI(
        api_key="sk-test",
        base_url="https://api.respan.ai",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    since = datetime(2026, 4, 9, tzinfo=UTC)
    traces = list(api.pull(since=since, limit=50))
    assert [t.trace_id for t in traces] == ["abc"]
    assert len(requests) == 1
    req = requests[0]
    assert req.url.path.endswith("/api/request-logs/list")
    assert req.headers["Authorization"] == "Bearer sk-test"
    assert "start_time" in dict(req.url.params)
    assert dict(req.url.params)["limit"] == "50"


def test_respan_api_raises_on_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    api = RespanAPI(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        list(api.pull())


def test_parse_relative_duration() -> None:
    assert parse_relative_duration("30m") == timedelta(minutes=30)
    assert parse_relative_duration("24h") == timedelta(hours=24)
    assert parse_relative_duration("7d") == timedelta(days=7)
    assert parse_relative_duration("2w") == timedelta(weeks=2)


def test_parse_relative_duration_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="unknown duration unit"):
        parse_relative_duration("5x")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_relative_duration("abch")


def test_trace_looks_like_failure_heuristics() -> None:
    good = Trace(trace_id="1", timestamp=datetime.now(tz=UTC), user_text="u", assistant_text="a")
    assert not good.looks_like_failure

    err = Trace(trace_id="1", timestamp=datetime.now(tz=UTC), user_text="u", assistant_text="", status="error")
    assert err.looks_like_failure

    tagged = Trace(trace_id="1", timestamp=datetime.now(tz=UTC), user_text="u", assistant_text="a", tags=["failure"])
    assert tagged.looks_like_failure
