"""Eval dataset loading.

An Arena dataset is a list of `EvalCase`s. Cases are typically stored as
JSONL on disk — one object per line with `id`, `inputs`, optional
`expected`, and optional `tags` / `source` / `trace_id` fields.

We keep the loader deliberately narrow: JSONL in, iterable out. Mining
from Respan traces produces JSONL too, so `arena mine` and `arena run`
share the same data format.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalCase:
    """One labelled case in an eval set."""

    id: str
    inputs: dict[str, Any]
    expected: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "handwritten"
    trace_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalCase:
        return cls(
            id=str(raw.get("id") or _auto_id(raw)),
            inputs=dict(raw.get("inputs") or {}),
            expected=dict(raw["expected"]) if raw.get("expected") is not None else None,
            tags=list(raw.get("tags") or []),
            source=str(raw.get("source") or "handwritten"),
            trace_id=raw.get("trace_id"),
        )

    def user_text(self) -> str:
        """Best-effort extraction of the user message for the gateway call.

        Conventions (in priority order):
          1. `inputs["prompt"]` — if caller already built the message.
          2. `inputs["ticket"]` — support-triage convention.
          3. `inputs["question"]` — Q&A convention.
          4. Falls back to `json.dumps(inputs)` so nothing is silently lost.
        """
        for key in ("prompt", "ticket", "question", "input", "text"):
            value = self.inputs.get(key)
            if isinstance(value, str) and value:
                return value
        return json.dumps(self.inputs, ensure_ascii=False)


@dataclass(slots=True)
class Dataset:
    """A lazy-ish eval dataset. Cases are held in memory once loaded."""

    name: str
    cases: list[EvalCase]

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> EvalCase:
        return self.cases[idx]

    def head(self, n: int) -> Dataset:
        return Dataset(name=self.name, cases=self.cases[:n])

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> Dataset:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"dataset not found: {p}")
        cases: list[EvalCase] = []
        with p.open() as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p}:{i}: invalid JSON — {exc}") from exc
                cases.append(EvalCase.from_dict(raw))
        return cls(name=name or p.stem, cases=cases)

    @classmethod
    def from_cases(cls, cases: Iterable[EvalCase], *, name: str = "memory") -> Dataset:
        return cls(name=name, cases=list(cases))


def _auto_id(raw: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(raw.get("inputs") or raw, sort_keys=True, default=str)
    return "case-" + hashlib.sha1(payload.encode()).hexdigest()[:10]
