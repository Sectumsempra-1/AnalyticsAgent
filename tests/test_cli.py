import json
from pathlib import Path

import pytest

import analytics_agent.graph as graph_module
from analytics_agent.cli import main, render
from analytics_agent.schemas import AnalysisResult, Finding
from tests.conftest import SALES_AMOUNT_SUM
from tests.fake_model import ScriptedModel, tool_call

SQL = "SELECT SUM(amount) AS total FROM sales"
FINDING = Finding(statement="Total sales are 1395.5", evidence="total = 1395.5", method="SQL SUM", code=SQL)


@pytest.fixture
def model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ScriptedModel:
    monkeypatch.setenv("MODEL_NAME", "scripted")
    monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path / "outputs"))
    scripted = ScriptedModel(
        turns=[tool_call("run_sql", query=SQL)],
        reports=[AnalysisResult(summary=f"Total sales are {SALES_AMOUNT_SUM}.", findings=[FINDING], confidence="high")],
    )
    monkeypatch.setattr(graph_module, "load_model", lambda settings: scripted)
    return scripted


def test_ask_prints_readable_answer(
    model: ScriptedModel, fixture_files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["ask", "What is the total?", str(fixture_files["sales"]), "--show-code"])
    out, err = capsys.readouterr()
    assert code == 0
    assert "Answer\n------\nTotal sales are 1395.5." in out
    assert "Confidence: HIGH" in out
    assert "1. Total sales are 1395.5" in out and "Evidence: total = 1395.5" in out
    assert SQL in out  # --show-code
    assert f"run_sql: {SQL}" in err  # live progress goes to stderr


def test_ask_json_output(model: ScriptedModel, fixture_files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", "What is the total?", str(fixture_files["sales"]), "--json"])
    out, err = capsys.readouterr()
    assert code == 0 and err == ""
    assert AnalysisResult.model_validate(json.loads(out)).findings == [FINDING]


def test_context_and_max_steps_reach_the_agent(
    model: ScriptedModel, fixture_files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ask", "Total?", str(fixture_files["sales"]), "--context", "amounts are in SAR", "--max-steps", "3"])
    system = str(model.seen[0][0].content)
    assert "amounts are in SAR" in system and "at most 3 turns" in system


def test_unsupported_file_type_is_rejected(model: ScriptedModel, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "q", "notes.txt"]) == 2
    err = capsys.readouterr().err
    assert "unsupported file type '.txt'" in err and "Value error" not in err
    assert model.seen == []


def test_missing_model_setting_is_reported(
    fixture_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("analytics_agent.config.load_dotenv", lambda: None)  # ignore the developer's .env
    monkeypatch.delenv("MODEL_NAME", raising=False)
    assert main(["ask", "q", str(fixture_files["sales"])]) == 2
    assert "MODEL_NAME is not set" in capsys.readouterr().err


def test_model_failure_is_reported_not_raised(
    fixture_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MODEL_NAME", "scripted")

    def unreachable(settings: object) -> None:
        raise ConnectionError("model server is not running")

    monkeypatch.setattr(graph_module, "load_model", unreachable)
    assert main(["ask", "q", str(fixture_files["sales"])]) == 2
    assert "model server is not running" in capsys.readouterr().err


def test_render_lists_sections_and_skips_empty_ones() -> None:
    result = AnalysisResult(
        summary="North leads.", data_quality_notes=["2 null regions"], limitations=["small data"], confidence="low"
    )
    text = render(result)
    assert "Data quality\n------------\n- 2 null regions" in text
    assert "Limitations" in text and "Findings" not in text and "Charts" not in text
