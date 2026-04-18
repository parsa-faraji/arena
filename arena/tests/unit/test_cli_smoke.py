from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from arena.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "arena" in result.stdout


def test_init_scaffolds(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.stdout
    assert (target / "arena.config.yaml").exists()
    assert (target / "prompts" / "v0-baseline.md").exists()
    assert (target / "dataset.jsonl").exists()
    assert (target / ".env").exists()


def test_run_without_api_key_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    result = runner.invoke(app, ["run", "v0", "--hello-world"])
    assert result.exit_code == 2
    assert "RESPAN_API_KEY" in result.stdout
