"""Project config (`arena.config.yaml`) schema + loader.

We parse the YAML with pydantic rather than raw dicts so typos fail loud
at load time. Silently skipping an unknown `type: exact_match_field` was
the old behaviour — that's exactly the kind of "works on my machine"
that bites you right before a demo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from arena.evals.evaluators import (
    Evaluator,
    ExactMatchEvaluator,
    JSONParseEvaluator,
    RegexEvaluator,
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactMatchSpec(_Base):
    type: Literal["exact_match"]
    field: str
    name: str | None = None


class JSONParseSpec(_Base):
    type: Literal["json_parse"]
    fields: list[str]
    name: str | None = None


class RegexSpec(_Base):
    type: Literal["regex"]
    pattern: str
    name: str | None = None
    flags: int = 0


class JudgeSpec(_Base):
    type: Literal["judge"]
    judge: str  # e.g. "pairwise", "rubric:helpfulness"
    field: str | None = None  # sub-field of output to judge (for rubric)
    name: str | None = None


EvaluatorSpec = Annotated[
    ExactMatchSpec | JSONParseSpec | RegexSpec | JudgeSpec,
    Field(discriminator="type"),
]


class FallbackLink(_Base):
    model: str
    fallbacks: list[str] = Field(default_factory=list)


class ProjectConfig(_Base):
    dataset: str = "dataset.jsonl"
    default_model: str = "gpt-4o-mini"
    judge_model: str = "claude-haiku-4-5"
    optimizer_model: str | None = None
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)
    judges: list[str] = Field(default_factory=list)
    fallback_chain: list[FallbackLink] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ProjectConfigError(
                f"{p}: must be a YAML mapping, got {type(data).__name__}"
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ProjectConfigError(f"{p}: {_format_validation_error(exc)}") from exc

    def to_evaluators(self) -> list[Evaluator]:
        """Instantiate the evaluator list. Judge evaluators are wired in Day 5."""
        out: list[Evaluator] = []
        for spec in self.evaluators:
            if isinstance(spec, ExactMatchSpec):
                out.append(ExactMatchEvaluator(field=spec.field, name=spec.name or ""))
            elif isinstance(spec, JSONParseSpec):
                out.append(
                    JSONParseEvaluator(
                        required_fields=tuple(spec.fields),
                        name=spec.name or "json_parse",
                    )
                )
            elif isinstance(spec, RegexSpec):
                out.append(
                    RegexEvaluator(
                        pattern=spec.pattern,
                        name=spec.name or "",
                        flags=spec.flags,
                    )
                )
            elif isinstance(spec, JudgeSpec):
                # Wired in Week-1 Day 5 once arena.judges lands. Until then
                # we carry the spec forward without instantiating so the
                # config parses cleanly.
                continue
        return out


class ProjectConfigError(ValueError):
    """Raised when arena.config.yaml is malformed."""


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"  {loc}: {err['msg']}")
    return "invalid config:\n" + "\n".join(lines)
