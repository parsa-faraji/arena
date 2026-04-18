from __future__ import annotations

from arena.evals.dataset import EvalCase
from arena.evals.evaluators import (
    ExactMatchEvaluator,
    JSONParseEvaluator,
    RegexEvaluator,
    parse_json_output,
)


def test_parse_json_output_direct() -> None:
    assert parse_json_output('{"urgency": "high"}') == {"urgency": "high"}


def test_parse_json_output_fenced() -> None:
    assert parse_json_output('```json\n{"x": 1}\n```') == {"x": 1}


def test_parse_json_output_embedded() -> None:
    assert parse_json_output('blah {"x": 1} trailing') == {"x": 1}


def test_parse_json_output_none() -> None:
    assert parse_json_output("definitely not json") is None


def test_exact_match_passes_with_json_output() -> None:
    case = EvalCase(id="c", inputs={}, expected={"urgency": "high"})
    ev = ExactMatchEvaluator(field="urgency")
    r = ev.score(case, '{"urgency": "HIGH", "other": 1}')
    assert r.passed is True
    assert r.score == 1.0


def test_exact_match_fails_when_missing() -> None:
    case = EvalCase(id="c", inputs={}, expected={"urgency": "high"})
    ev = ExactMatchEvaluator(field="urgency")
    r = ev.score(case, '{"category": "bug"}')
    assert r.passed is False
    assert r.score == 0.0


def test_exact_match_skips_when_no_expected() -> None:
    case = EvalCase(id="c", inputs={})
    ev = ExactMatchEvaluator(field="urgency")
    r = ev.score(case, '{"urgency": "high"}')
    assert r.passed is False
    assert r.details.get("skipped")


def test_json_parse_evaluator() -> None:
    case = EvalCase(id="c", inputs={})
    ev = JSONParseEvaluator(required_fields=("urgency", "category"))
    ok = ev.score(case, '{"urgency": "high", "category": "billing"}')
    assert ok.passed
    missing = ev.score(case, '{"urgency": "high"}')
    assert not missing.passed
    assert missing.details["missing"] == ["category"]
    bad = ev.score(case, "not json")
    assert not bad.passed
    assert bad.details.get("parse_error") is True


def test_regex_evaluator() -> None:
    case = EvalCase(id="c", inputs={})
    ev = RegexEvaluator(pattern=r"refund")
    assert ev.score(case, "I'll refund you today").passed
    assert not ev.score(case, "no match here").passed
