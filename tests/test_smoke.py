"""End-to-end run against the real model configured in .env. Run with: pytest -m smoke"""

import json
import urllib.request
from pathlib import Path

import pytest

from analytics_agent.config import ConfigError, Settings
from analytics_agent.graph import build_agent
from analytics_agent.schemas import AnalysisRequest, AnalysisResult

pytestmark = pytest.mark.smoke


def _settings_or_skip(tmp_path: Path) -> Settings:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        pytest.skip(f"model not configured: {exc}")
    if settings.model_provider == "ollama":
        base = settings.model_base_url or "http://127.0.0.1:11434"
        try:
            urllib.request.urlopen(f"{base}/api/version", timeout=3)
        except OSError:
            pytest.skip("Ollama is not running")
    return Settings(**{**settings.__dict__, "output_dir": tmp_path / "outputs"})


def test_end_to_end_with_real_model(tmp_path: Path, fixture_files: dict[str, Path]) -> None:
    settings = _settings_or_skip(tmp_path)
    request = AnalysisRequest(
        question="Which region has the highest total sales amount, and what is that total?",
        dataset_paths=[str(fixture_files["sales"])],
        max_steps=8,
    )
    out = build_agent(settings=settings).invoke(request.model_dump())
    result = AnalysisResult.model_validate(out)
    print(json.dumps(result.model_dump(), indent=2))

    # Planted answer: north = 100 + 300 + 220 + 100 = 720 (the duplicate row is counted).
    text = json.dumps(result.model_dump()).lower()
    assert "north" in text
    assert result.findings, "expected at least one verified finding"
