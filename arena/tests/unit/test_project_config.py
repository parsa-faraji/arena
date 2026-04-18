from __future__ import annotations

from pathlib import Path

import pytest

from arena.evals.evaluators import (
    ExactMatchEvaluator,
    JSONParseEvaluator,
    RegexEvaluator,
)
from arena.project import ProjectConfig, ProjectConfigError


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = ProjectConfig.from_yaml(tmp_path / "does-not-exist.yaml")
    assert cfg.default_model == "gpt-4o-mini"
    assert cfg.evaluators == []


def test_valid_yaml_parses(tmp_path: Path) -> None:
    cfg = ProjectConfig.from_yaml(
        _write(
            tmp_path / "a.yaml",
            """
dataset: dataset.jsonl
default_model: claude-haiku-4-5
evaluators:
  - type: exact_match
    field: urgency
  - type: json_parse
    fields: [urgency, category]
  - type: regex
    pattern: "refund"
""",
        )
    )
    assert cfg.default_model == "claude-haiku-4-5"
    evals = cfg.to_evaluators()
    assert isinstance(evals[0], ExactMatchEvaluator)
    assert isinstance(evals[1], JSONParseEvaluator)
    assert isinstance(evals[2], RegexEvaluator)


def test_unknown_evaluator_type_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ProjectConfigError, match="evaluators"):
        ProjectConfig.from_yaml(
            _write(
                tmp_path / "a.yaml",
                "evaluators:\n  - type: exact_match_field\n    field: urgency\n",
            )
        )


def test_typo_in_key_rejected(tmp_path: Path) -> None:
    # `feild` instead of `field` on exact_match
    with pytest.raises(ProjectConfigError):
        ProjectConfig.from_yaml(
            _write(
                tmp_path / "a.yaml",
                "evaluators:\n  - type: exact_match\n    feild: urgency\n",
            )
        )


def test_non_mapping_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProjectConfigError, match="mapping"):
        ProjectConfig.from_yaml(_write(tmp_path / "a.yaml", "- just\n- a\n- list\n"))


def test_judge_spec_needs_client_to_instantiate(tmp_path: Path) -> None:
    cfg = ProjectConfig.from_yaml(
        _write(
            tmp_path / "a.yaml",
            "evaluators:\n"
            "  - type: judge\n"
            "    judge: rubric\n"
            "    criterion: 'is it helpful?'\n"
            "    name: helpfulness\n",
        )
    )
    assert len(cfg.evaluators) == 1
    # Without a client, judges are skipped — this keeps `arena run` without
    # credentials from crashing and lets the dashboard load configs read-only.
    assert cfg.to_evaluators() == []


def test_rubric_judge_requires_criterion(tmp_path: Path) -> None:
    with pytest.raises(ProjectConfigError, match="criterion"):
        ProjectConfig.from_yaml(
            _write(
                tmp_path / "a.yaml",
                "evaluators:\n  - type: judge\n    judge: rubric\n",
            )
        )


def test_ensemble_judge_requires_nested_judges(tmp_path: Path) -> None:
    with pytest.raises(ProjectConfigError, match="nested"):
        ProjectConfig.from_yaml(
            _write(
                tmp_path / "a.yaml",
                "evaluators:\n  - type: judge\n    judge: ensemble\n",
            )
        )


def test_judge_evaluator_built_when_client_supplied(tmp_path: Path) -> None:
    from arena.evals.evaluators import JudgeEvaluator

    cfg = ProjectConfig.from_yaml(
        _write(
            tmp_path / "a.yaml",
            "evaluators:\n"
            "  - type: judge\n"
            "    judge: rubric\n"
            "    criterion: 'reply quality'\n"
            "    target_field: suggested_reply\n"
            "    name: helpfulness\n",
        )
    )

    class _StubClient:
        pass

    evals = cfg.to_evaluators(client=_StubClient())  # type: ignore[arg-type]
    assert len(evals) == 1
    assert isinstance(evals[0], JudgeEvaluator)
    assert evals[0].name == "helpfulness"
