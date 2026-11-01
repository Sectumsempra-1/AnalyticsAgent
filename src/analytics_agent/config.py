"""Runtime settings read from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

STRUCTURED_OUTPUT_METHODS = ("function_calling", "json_schema", "json_mode")


class ConfigError(ValueError):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Everything configurable about a run. Build with Settings.from_env() or directly in tests."""

    model_name: str
    model_provider: str | None = None
    model_base_url: str | None = None
    model_temperature: float = 0.0
    structured_output_method: str = "function_calling"
    output_dir: Path = Path("outputs")
    run_timeout_s: float = 600.0
    tool_timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings from the environment. Values already set take precedence over .env."""
        load_dotenv()
        name = os.getenv("MODEL_NAME", "").strip()
        if not name:
            raise ConfigError("MODEL_NAME is not set (see .env.example)")
        method = os.getenv("MODEL_STRUCTURED_OUTPUT_METHOD", "function_calling")
        if method not in STRUCTURED_OUTPUT_METHODS:
            raise ConfigError(f"MODEL_STRUCTURED_OUTPUT_METHOD must be one of {STRUCTURED_OUTPUT_METHODS}")
        return cls(
            model_name=name,
            model_provider=os.getenv("MODEL_PROVIDER") or None,
            model_base_url=os.getenv("MODEL_BASE_URL") or None,
            model_temperature=_float("MODEL_TEMPERATURE", 0.0),
            structured_output_method=method,
            output_dir=Path(os.getenv("AGENT_OUTPUT_DIR", "outputs")),
            run_timeout_s=_float("AGENT_RUN_TIMEOUT_S", 600.0),
            tool_timeout_s=_float("AGENT_TOOL_TIMEOUT_S", 30.0),
        )


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if key != "MODEL_TEMPERATURE" and value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value
