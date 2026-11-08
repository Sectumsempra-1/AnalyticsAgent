from pathlib import Path

import pytest

from analytics_agent.config import ConfigError, Settings

ENV_KEYS = [
    "MODEL_NAME", "MODEL_PROVIDER", "MODEL_BASE_URL", "MODEL_TEMPERATURE", "MODEL_CONTEXT_TOKENS",
    "MODEL_STRUCTURED_OUTPUT_METHOD", "AGENT_OUTPUT_DIR", "AGENT_RUN_TIMEOUT_S", "AGENT_TOOL_TIMEOUT_S",
]  # fmt: skip


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty project folder as the working directory, with no settings in the environment."""
    for key in ENV_KEYS:  # setenv first so monkeypatch restores keys that .env loading adds
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_reads_dotenv_from_working_directory(project: Path) -> None:
    (project / ".env").write_text(
        "MODEL_PROVIDER=ollama\nMODEL_NAME=from-dotenv\nMODEL_CONTEXT_TOKENS=8192\nAGENT_TOOL_TIMEOUT_S=5\n"
    )
    settings = Settings.from_env()
    assert (settings.model_provider, settings.model_name) == ("ollama", "from-dotenv")
    assert (settings.context_tokens, settings.tool_timeout_s) == (8192, 5.0)


def test_reads_dotenv_from_parent_directory(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / ".env").write_text("MODEL_NAME=from-parent\n")
    (project / "sub").mkdir()
    monkeypatch.chdir(project / "sub")
    assert Settings.from_env().model_name == "from-parent"


def test_environment_overrides_dotenv(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / ".env").write_text("MODEL_NAME=from-dotenv\n")
    monkeypatch.setenv("MODEL_NAME", "from-env")
    assert Settings.from_env().model_name == "from-env"


def test_defaults(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "m")
    settings = Settings.from_env()
    assert settings == Settings(model_name="m")


def test_missing_model_name(project: Path) -> None:
    with pytest.raises(ConfigError, match="MODEL_NAME"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("MODEL_STRUCTURED_OUTPUT_METHOD", "xml", "must be one of"),
        ("AGENT_RUN_TIMEOUT_S", "ten", "must be a number"),
        ("AGENT_TOOL_TIMEOUT_S", "0", "must be positive"),
    ],
)
def test_invalid_values(project: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str, message: str) -> None:
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigError, match=message):
        Settings.from_env()
