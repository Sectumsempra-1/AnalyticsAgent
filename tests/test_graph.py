from pathlib import Path
from typing import Any, TypedDict

import pytest
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from analytics_agent.config import Settings
from analytics_agent.graph import build_agent
from analytics_agent.schemas import AnalysisRequest, AnalysisResult, Finding
from tests.conftest import SALES_AMOUNT_SUM
from tests.fake_model import ScriptedModel, tool_call

SUM_SQL = "SELECT SUM(amount) AS total FROM sales"


def report(*findings: Finding, confidence: str = "high", summary: str = "Total sales are 1395.5.") -> AnalysisResult:
    return AnalysisResult(summary=summary, findings=list(findings), confidence=confidence)  # type: ignore[arg-type]


TOTAL_FINDING = Finding(
    statement="Total sales amount is 1395.5", evidence="total = 1395.5", method="SQL SUM", code=SUM_SQL
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(model_name="scripted", output_dir=tmp_path / "outputs", tool_timeout_s=20)


def run(model: ScriptedModel, settings: Settings, files: dict[str, Path], **request: Any) -> dict[str, Any]:
    payload = {"question": "What is the total sales amount?", "dataset_paths": [str(files["sales"])], **request}
    out: dict[str, Any] = build_agent(model, settings).invoke(payload)
    return out


def test_happy_path(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(
        turns=[tool_call("profile_dataset", "c1", table="sales"), tool_call("run_sql", "c2", query=SUM_SQL)],
        reports=[report(TOTAL_FINDING)],
    )
    out = run(model, settings, fixture_files)

    assert set(out) == set(AnalysisResult.model_fields)  # output matches the contract exactly
    result = AnalysisResult.model_validate(out)
    assert result.findings == [TOTAL_FINDING]
    assert result.confidence == "high"
    assert model.bound_tools == ["profile_dataset", "run_sql", "run_python", "make_chart"]
    # the model saw real tool output, including the planted-duplicate profile line
    tool_texts = [m.content for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert any("Duplicate rows: 1" in t for t in tool_texts)
    assert any(str(SALES_AMOUNT_SUM) in t for t in tool_texts)


def test_system_prompt_lists_tables_and_context(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(reports=[report(confidence="low")])
    run(model, settings, fixture_files, context="amount is in SAR")
    system = model.seen[0][0].content
    assert "- sales: 10 rows x 4 columns" in system
    assert "amount is in SAR" in system


def test_invented_number_is_removed(settings: Settings, fixture_files: dict[str, Path]) -> None:
    invented = Finding(statement="Sales grew 37%", evidence="growth = 37%", method="SQL", code=SUM_SQL)
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=SUM_SQL)], reports=[report(TOTAL_FINDING, invented)]
    )
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert result.findings == [TOTAL_FINDING]
    assert result.confidence == "medium"
    assert any("Sales grew 37%" in note for note in result.limitations)


def test_step_limit_stops_the_loop(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(
        turns=[tool_call("run_sql", f"c{i}", query="SELECT 1") for i in range(10)],
        reports=[report(confidence="medium")],
    )
    result = AnalysisResult.model_validate(run(model, settings, fixture_files, max_steps=2))
    assert len(model.seen) == 2  # exactly two agent turns
    assert any("step limit (2)" in note for note in result.limitations)


def test_total_time_limit_stops_the_loop(tmp_path: Path, fixture_files: dict[str, Path]) -> None:
    settings = Settings(model_name="scripted", output_dir=tmp_path, run_timeout_s=0.001)
    model = ScriptedModel(turns=[tool_call("run_sql", query="SELECT 1")], reports=[report(confidence="low")])
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert len(model.seen) == 0  # deadline passed during dataset loading
    assert any("total time limit" in note for note in result.limitations)


def test_unknown_tool_is_reported_to_the_model(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(turns=[tool_call("delete_everything")], reports=[report(confidence="low")])
    run(model, settings, fixture_files)
    tool_msg = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
    assert "unknown tool 'delete_everything'" in str(tool_msg.content)


def test_bad_tool_arguments_are_reported(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(turns=[tool_call("run_sql", sql="SELECT 1")], reports=[report(confidence="low")])
    run(model, settings, fixture_files)
    tool_msg = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
    assert "invalid arguments for run_sql" in str(tool_msg.content)


def test_invalid_report_is_retried(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=SUM_SQL)], reports=[ValueError("missing summary"), report(TOTAL_FINDING)]
    )
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert result.findings == [TOTAL_FINDING]


def test_report_failing_every_attempt_gives_low_confidence_fallback(
    settings: Settings, fixture_files: dict[str, Path]
) -> None:
    model = ScriptedModel(reports=[ValueError("bad"), "plain text", ValueError("still bad")])
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert result.confidence == "low"
    assert "could not produce a valid structured report" in result.summary
    assert [m for m, _ in model.report_calls] == ["function_calling", "function_calling", "json_schema"]


def test_plain_text_reply_gets_a_specific_retry_prompt(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=SUM_SQL)], reports=["Total sales are 1395.5.", report(TOTAL_FINDING)]
    )
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert result.findings == [TOTAL_FINDING]
    retry = model.report_calls[1][1][-1].content
    assert "plain text" in retry and "AnalysisResult tool" in retry


def test_report_written_as_json_text_is_accepted(settings: Settings, fixture_files: dict[str, Path]) -> None:
    text = "Here is the report:\n```json\n" + report(TOTAL_FINDING).model_dump_json() + "\n```"
    model = ScriptedModel(turns=[tool_call("run_sql", query=SUM_SQL)], reports=[text])
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert result.findings == [TOTAL_FINDING]
    assert len(model.report_calls) == 1


def test_strict_last_attempt_uses_a_fresh_context(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=SUM_SQL)],
        reports=["no tool call", "no tool call again", report(TOTAL_FINDING)],
    )
    out = build_agent(model, settings).stream(
        {"question": "Total?", "dataset_paths": [str(fixture_files["sales"])]}, stream_mode="values"
    )
    state = list(out)[-1]
    assert state["findings"] == [TOTAL_FINDING.model_dump()]
    method, messages = model.report_calls[2]
    assert method == "json_schema"
    assert "plain text" not in str(messages[-1].content)  # earlier retry prompts were dropped


def test_provider_error_during_report_is_retried(settings: Settings, fixture_files: dict[str, Path]) -> None:
    # e.g. Ollama's ResponseError for a truncated tool call; must not crash the run
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=SUM_SQL)],
        reports=[RuntimeError("error parsing tool call: unexpected end of JSON input"), report(TOTAL_FINDING)],
    )
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert result.findings == [TOTAL_FINDING]
    assert "unexpected end of JSON input" in str(model.report_calls[1][1][-1].content)


def test_charts_come_from_saved_files(settings: Settings, fixture_files: dict[str, Path]) -> None:
    model = ScriptedModel(
        turns=[tool_call("make_chart", code="sales['amount'].plot()")],
        reports=[report(confidence="low").model_copy(update={"charts": ["made_up.png"]})],
    )
    result = AnalysisResult.model_validate(run(model, settings, fixture_files))
    assert len(result.charts) == 1
    assert result.charts[0].endswith("chart_01.png") and Path(result.charts[0]).exists()


def test_missing_dataset_returns_low_confidence_result(settings: Settings, tmp_path: Path) -> None:
    model = ScriptedModel()
    out = build_agent(model, settings).invoke({"question": "q", "dataset_paths": [str(tmp_path / "nope.csv")]})
    result = AnalysisResult.model_validate(out)
    assert result.confidence == "low"
    assert "dataset not found" in result.limitations[0]
    assert model.seen == []


def test_malformed_request_raises(settings: Settings, fixture_files: dict[str, Path]) -> None:
    with pytest.raises(ValidationError):
        run(ScriptedModel(), settings, fixture_files, question="   ")


def test_published_schemas_match_contract(settings: Settings) -> None:
    graph = build_agent(ScriptedModel(), settings)
    assert set(graph.get_input_jsonschema()["properties"]) == set(AnalysisRequest.model_fields)
    assert set(graph.get_output_jsonschema()["properties"]) == set(AnalysisResult.model_fields)


def test_works_as_subgraph_of_a_researcher(settings: Settings, fixture_files: dict[str, Path]) -> None:
    analyst = build_agent(
        ScriptedModel(turns=[tool_call("run_sql", query=SUM_SQL)], reports=[report(TOTAL_FINDING)]), settings
    )

    class ResearcherState(TypedDict, total=False):
        question: str
        files: list[str]
        analysis: dict[str, Any]

    def call_analyst(state: ResearcherState) -> dict[str, Any]:
        request = AnalysisRequest(question=state["question"], dataset_paths=state["files"])
        return {"analysis": analyst.invoke(request.model_dump())}

    parent = StateGraph(ResearcherState)
    parent.add_node("analyst", call_analyst)
    parent.add_edge(START, "analyst")
    parent.add_edge("analyst", END)
    out = parent.compile().invoke({"question": "Total?", "files": [str(fixture_files["sales"])]})
    assert AnalysisResult.model_validate(out["analysis"]).findings == [TOTAL_FINDING]
