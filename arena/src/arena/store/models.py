"""SQLModel tables for Arena runs, variants, cases, scores, and optimizer steps.

The schema is intentionally small enough to fit on one screen — each row is
cheap and we'd rather have too many runs than a tangled schema.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel
from sqlmodel import create_engine as _sqlmodel_create_engine


def _uid() -> str:
    return uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(tz=UTC)


class Variant(SQLModel, table=True):
    """A named prompt+model configuration."""

    id: str = Field(default_factory=_uid, primary_key=True)
    name: str = Field(index=True)
    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Case(SQLModel, table=True):
    """One eval case. `inputs` and `expected` are JSON-encoded strings."""

    id: str = Field(default_factory=_uid, primary_key=True)
    dataset: str = Field(index=True)
    inputs_json: str
    expected_json: str | None = None
    tags_json: str = "[]"
    source: str = "handwritten"  # "handwritten" | "respan-trace" | "synthetic"
    trace_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=_now)

    @property
    def inputs(self) -> dict[str, Any]:
        return json.loads(self.inputs_json)

    @property
    def expected(self) -> dict[str, Any] | None:
        return json.loads(self.expected_json) if self.expected_json else None

    @property
    def tags(self) -> list[str]:
        return json.loads(self.tags_json)


class Run(SQLModel, table=True):
    """One execution of a variant across a dataset."""

    id: str = Field(default_factory=_uid, primary_key=True)
    variant_id: str = Field(foreign_key="variant.id", index=True)
    dataset: str = Field(index=True)
    status: str = "pending"  # "pending" | "running" | "done" | "error"
    total_cases: int = 0
    completed_cases: int = 0
    error: str | None = None
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None


class CaseResult(SQLModel, table=True):
    """One model output for one case in one run."""

    id: str = Field(default_factory=_uid, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    case_id: str = Field(foreign_key="case.id", index=True)
    output: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str
    cache_hit: bool = False
    error: str | None = None


class JudgeScore(SQLModel, table=True):
    """One judge's score for one case result."""

    id: str = Field(default_factory=_uid, primary_key=True)
    result_id: str = Field(foreign_key="caseresult.id", index=True)
    judge: str  # e.g., "pairwise", "rubric:helpfulness"
    score: float
    rationale: str | None = None
    raw_json: str | None = None


class OptimizerStep(SQLModel, table=True):
    """One proposal in an optimizer run."""

    id: str = Field(default_factory=_uid, primary_key=True)
    parent_variant_id: str = Field(foreign_key="variant.id", index=True)
    proposed_variant_id: str = Field(foreign_key="variant.id", index=True)
    step: int
    gradient_text: str | None = None
    score_before: float | None = None
    score_after: float | None = None
    kept: bool = False
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------- engine

def create_engine(url: str) -> Engine:
    """Create a SQLModel engine. SQLite gets the usual check_same_thread tweak."""
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return _sqlmodel_create_engine(url, connect_args=connect_args, echo=False)


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as s:
        yield s
