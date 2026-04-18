from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.evals.dataset import Dataset, EvalCase


def test_from_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "inputs": {"ticket": "hello"}, "expected": {"category": "other"}}),
                "# comment lines are skipped",
                "",
                json.dumps({"inputs": {"ticket": "refund please"}, "expected": {"urgency": "high"}}),
            ]
        )
    )
    ds = Dataset.from_jsonl(p)
    assert ds.name == "cases"
    assert len(ds) == 2
    first = ds[0]
    assert first.id == "a"
    assert first.expected == {"category": "other"}
    # second case gets an auto-generated id
    assert ds[1].id.startswith("case-")


def test_user_text_fallbacks() -> None:
    assert EvalCase(id="x", inputs={"ticket": "hi"}).user_text() == "hi"
    assert EvalCase(id="x", inputs={"question": "why?"}).user_text() == "why?"
    assert EvalCase(id="x", inputs={"other": "nothing obvious"}).user_text().startswith("{")


def test_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Dataset.from_jsonl(tmp_path / "nope.jsonl")


def test_bad_json_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a"}\n{not json}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        Dataset.from_jsonl(p)


def test_head() -> None:
    cases = [EvalCase(id=str(i), inputs={"text": str(i)}) for i in range(5)]
    ds = Dataset.from_cases(cases, name="demo")
    assert len(ds.head(2)) == 2
    assert ds.head(2).cases[1].id == "1"
