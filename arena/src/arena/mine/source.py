"""Trace sources — where the miner gets its raw data.

The core loop of the miner is: pull traces, cluster, label, emit eval cases.
`TraceSource` is the seam for "pull traces". We ship two implementations:

- `RespanAPI` — REST client for a Respan workspace. The exact endpoint +
  response shape aren't locked in the public docs yet, so the path is a
  constructor arg and the JSON-to-`Trace` adapter lives in one method
  (`_trace_from_row`) that's easy to tweak against a real account.

- `FixtureSource` — loads traces from a local JSONL file. This is what
  the demo uses so Arena works end-to-end without network or
  credentials.

Both yield lightweight `Trace` dataclasses, which is what the rest of
`arena.mine` consumes. Downstream code never sees the wire format.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Trace:
    """Normalised view of one LLM call pulled from a trace source."""

    trace_id: str
    timestamp: datetime
    user_text: str
    assistant_text: str
    model: str = ""
    latency_ms: int = 0
    status: str = "success"  # "success" | "error" | "flagged"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def looks_like_failure(self) -> bool:
        """Cheap heuristic for the 'this looks like a problem' signal."""
        if self.status != "success":
            return True
        if "failure" in self.tags or "regression" in self.tags:
            return True
        return "error" in self.metadata


class TraceSource(Protocol):
    def pull(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
        only_failures: bool = False,
    ) -> Iterator[Trace]: ...


# --------------------------------------------------------------- fixture

class FixtureSource:
    """Loads traces from a local JSONL file.

    Used by the demo and by tests. Each line is a JSON object with the
    same shape as the Respan API response (see `_trace_from_row`), so
    fixtures can be recorded from a real account and replayed without
    any transformation.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"fixture not found: {self._path}")

    def pull(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
        only_failures: bool = False,
    ) -> Iterator[Trace]:
        count = 0
        with self._path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                trace = _trace_from_row(json.loads(line))
                if since and trace.timestamp < since:
                    continue
                if until and trace.timestamp > until:
                    continue
                if only_failures and not trace.looks_like_failure:
                    continue
                yield trace
                count += 1
                if count >= limit:
                    return


# -------------------------------------------------------------- respan api

class RespanAPI:
    """Minimal REST client for pulling traces from a Respan workspace.

    The public docs do not yet lock in the logs/traces endpoint URL, so
    it is a constructor argument. If Respan ships a different shape the
    `_trace_from_row` adapter is the only thing that needs to move.

    Authentication uses the same `RESPAN_API_KEY` as the gateway. We
    keep HTTP concerns (retries, timeouts) narrow: one GET, structured
    error, caller handles pagination via `since`/`until`.
    """

    DEFAULT_PATH = "/api/request-logs/list"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.respan.ai",
        path: str = DEFAULT_PATH,
        timeout_s: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._path = path
        self._timeout = timeout_s
        self._http = http_client

    def pull(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
        only_failures: bool = False,
    ) -> Iterator[Trace]:
        params: dict[str, Any] = {"limit": limit}
        if since is not None:
            params["start_time"] = since.isoformat()
        if until is not None:
            params["end_time"] = until.isoformat()
        if only_failures:
            params["status"] = "error"

        client = self._http or httpx.Client(timeout=self._timeout)
        try:
            resp = client.get(
                f"{self._base_url}{self._path}",
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            payload = resp.json()
        finally:
            if self._http is None:
                client.close()

        for row in _iter_rows(payload):
            try:
                yield _trace_from_row(row)
            except Exception as exc:
                log.warning("skipping malformed trace row: %s", exc)


# ------------------------------------------------------------- shared helpers

def _iter_rows(payload: Any) -> Iterator[dict[str, Any]]:
    """Accept a handful of shapes the API might hand back."""
    if isinstance(payload, list):
        yield from payload
        return
    if not isinstance(payload, dict):
        return
    for key in ("results", "data", "items", "traces", "logs"):
        if key in payload and isinstance(payload[key], list):
            yield from payload[key]
            return
    yield payload  # single object fallback


def _trace_from_row(row: dict[str, Any]) -> Trace:
    """Normalise a wire-format trace row into a `Trace`.

    Fields are looked up under multiple synonyms so the same adapter
    works against fixtures recorded at different Respan versions or
    against OTLP-style exports.
    """
    timestamp = _parse_ts(
        row.get("timestamp")
        or row.get("created_at")
        or row.get("start_time")
        or row.get("time")
    )
    messages = row.get("prompt_messages") or row.get("messages") or []
    user_text = _last_role_content(messages, "user") or str(row.get("prompt", ""))
    assistant_text = (
        _last_role_content(messages, "assistant")
        or row.get("completion_message", {}).get("content", "")
        if isinstance(row.get("completion_message"), dict)
        else str(row.get("completion", ""))
    )
    status = str(row.get("status") or ("error" if row.get("error") else "success"))
    return Trace(
        trace_id=str(row.get("id") or row.get("trace_id") or row.get("unique_id") or ""),
        timestamp=timestamp,
        user_text=user_text,
        assistant_text=str(assistant_text or ""),
        model=str(row.get("model", "")),
        latency_ms=int(row.get("latency_ms") or row.get("latency") or 0),
        status=status,
        tags=list(row.get("tags") or []),
        metadata=dict(row.get("metadata") or {}),
    )


def _parse_ts(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=UTC)
    if isinstance(raw, str) and raw:
        try:
            # handle trailing "Z"
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(tz=UTC)


def _last_role_content(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == role:
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
    return ""


def parse_relative_duration(text: str) -> timedelta:
    """Parse '24h', '7d', '30m' into a timedelta. Used by the CLI's --last."""
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    unit = text[-1]
    try:
        amount = int(text[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid duration {text!r}") from exc
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    raise ValueError(f"unknown duration unit {unit!r}; use m/h/d/w")
