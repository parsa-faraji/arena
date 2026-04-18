"""Configuration loaded from environment + .env files."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Arena runtime settings.

    Load order: process env → nearest .env file → defaults.
    Any CLI call instantiates this fresh so tests can monkeypatch env cleanly.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    respan_api_key: SecretStr | None = Field(default=None, alias="RESPAN_API_KEY")
    respan_org_id: str | None = Field(default=None, alias="RESPAN_ORG_ID")
    respan_base_url: str = Field(
        default="https://api.respan.ai/api",
        alias="RESPAN_BASE_URL",
    )

    default_model: str = Field(default="gpt-4o-mini", alias="ARENA_DEFAULT_MODEL")
    judge_model: str = Field(default="claude-haiku-4-5", alias="ARENA_JUDGE_MODEL")
    optimizer_model: str = Field(default="claude-sonnet-4-6", alias="ARENA_OPTIMIZER_MODEL")

    database_url: str = Field(
        default="sqlite:///arena.db",
        alias="DATABASE_URL",
    )
    disable_log: bool = Field(default=False, alias="ARENA_DISABLE_LOG")

    project_root: Path = Field(default_factory=Path.cwd)

    @property
    def has_respan_credentials(self) -> bool:
        return self.respan_api_key is not None

    def respan_api_key_value(self) -> str:
        if self.respan_api_key is None:
            raise RuntimeError(
                "RESPAN_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.respan_api_key.get_secret_value()
