"""Project config (`arena.config.yaml`) schema + loader.

We parse the YAML with pydantic rather than raw dicts so typos fail loud
at load time. Silently skipping an unknown `type: exact_match_field` was
the old behaviour — that's exactly the kind of "works on my machine"
that bites you right before a demo.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from arena.evals.evaluators import (
    Evaluator,
    ExactMatchEvaluator,
    JSONParseEvaluator,
    JudgeEvaluator,
    RegexEvaluator,
)

if TYPE_CHECKING:
    from arena.gateway.client import GatewayClient
    from arena.judges.base import Judge


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
    judge: Literal["rubric", "reference", "ensemble"] = "rubric"
    name: str | None = None
    model: str | None = None
    temperature: float = 0.0
    # rubric-specific
    criterion: str | None = None
    target_field: str | None = None
    # reference-specific
    reference_field: str = "reference"
    # ensemble-specific: nested rubric/reference judges
    judges: list[JudgeSpec] | None = None

    @model_validator(mode="after")
    def _validate_kind(self) -> JudgeSpec:
        if self.judge == "rubric" and not self.criterion:
            raise ValueError("rubric judge requires 'criterion'")
        if self.judge == "ensemble" and not self.judges:
            raise ValueError("ensemble judge requires 'judges' (nested list)")
        if self.judge != "ensemble" and self.judges:
            raise ValueError("'judges' is only valid for judge: ensemble")
        return self

    def build_judge(self, default_model: str) -> Judge:
        """Instantiate the concrete Judge implementation."""
        from arena.judges.ensemble import JudgeEnsemble
        from arena.judges.reference import ReferenceJudge
        from arena.judges.rubric import RubricJudge

        model = self.model or default_model
        if self.judge == "rubric":
            return RubricJudge(
                criterion=self.criterion or "",
                name=self.name or "",
                model=model,
                temperature=self.temperature,
                target_field=self.target_field,
            )
        if self.judge == "reference":
            return ReferenceJudge(
                reference_field=self.reference_field,
                name=self.name or "reference",
                model=model,
                temperature=self.temperature,
            )
        if self.judge == "ensemble":
            nested = [spec.build_judge(default_model) for spec in (self.judges or [])]
            return JudgeEnsemble(judges=nested, name=self.name or "ensemble")
        raise ValueError(f"unknown judge kind: {self.judge}")


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

    def to_evaluators(
        self,
        *,
        client: GatewayClient | None = None,
    ) -> list[Evaluator]:
        """Instantiate the evaluator list.

        If `client` is provided, judge evaluators are wired up. Otherwise
        they're skipped — useful for dry-running the config without a
        Respan API key (tests, docs, etc.).
        """
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
                if client is None:
                    continue
                judge = spec.build_judge(default_model=self.judge_model)
                out.append(JudgeEvaluator(judge, client, name=spec.name or None))
        return out


class ProjectConfigError(ValueError):
    """Raised when arena.config.yaml is malformed."""


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"  {loc}: {err['msg']}")
    return "invalid config:\n" + "\n".join(lines)
