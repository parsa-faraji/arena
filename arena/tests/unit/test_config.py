from __future__ import annotations

import pytest

from arena.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings()
    assert s.respan_base_url == "https://api.respan.ai/api"
    assert s.default_model == "gpt-4o-mini"
    assert s.database_url == "sqlite:///arena.db"
    assert s.has_respan_credentials is False


def test_api_key_required_for_access(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    s = Settings()
    with pytest.raises(RuntimeError, match="RESPAN_API_KEY"):
        s.respan_api_key_value()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESPAN_API_KEY", "sk-test-abc")
    s = Settings()
    assert s.has_respan_credentials is True
    assert s.respan_api_key_value() == "sk-test-abc"
