import pytest
from pydantic import ValidationError

from analytics_agent.schemas import AnalysisRequest, AnalysisResult, Finding


def make_finding(**overrides: str) -> Finding:
    fields = {
        "statement": "Revenue grew in Q2.",
        "evidence": "Q1=1,000; Q2=1,250 (+25.0%)",
        "method": "SQL SUM grouped by quarter",
        "code": "SELECT quarter, SUM(revenue) FROM sales GROUP BY quarter",
    }
    fields.update(overrides)
    return Finding(**fields)


def make_result(**overrides: object) -> AnalysisResult:
    fields: dict[str, object] = {"summary": "Revenue grew 25% in Q2.", "confidence": "high"}
    fields.update(overrides)
    return AnalysisResult(**fields)


# --- AnalysisRequest ---------------------------------------------------------


def test_request_defaults() -> None:
    req = AnalysisRequest(question="What drives churn?", dataset_paths=["data.csv"])
    assert req.max_steps == 15
    assert req.context is None


def test_request_accepts_all_supported_extensions_case_insensitive() -> None:
    paths = ["a.csv", "b.parquet", "c.xlsx", "D.CSV"]
    assert AnalysisRequest(question="q", dataset_paths=paths).dataset_paths == paths


@pytest.mark.parametrize("question", ["", "   "])
def test_request_rejects_blank_question(question: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(question=question, dataset_paths=["a.csv"])


def test_request_rejects_empty_dataset_list() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(question="q", dataset_paths=[])


@pytest.mark.parametrize("path", ["data.json", "data", "data.csv.bak"])
def test_request_rejects_unsupported_extension(path: str) -> None:
    with pytest.raises(ValidationError, match="unsupported file type"):
        AnalysisRequest(question="q", dataset_paths=[path])


@pytest.mark.parametrize("steps", [0, -1, 51])
def test_request_rejects_out_of_range_max_steps(steps: int) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(question="q", dataset_paths=["a.csv"], max_steps=steps)


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(question="q", dataset_paths=["a.csv"], maxsteps=5)  # type: ignore[call-arg]


# --- Finding -----------------------------------------------------------------


@pytest.mark.parametrize("field", ["statement", "evidence", "method", "code"])
def test_finding_rejects_blank_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        make_finding(**{field: "  "})


# --- AnalysisResult ----------------------------------------------------------


def test_result_defaults_are_empty_lists() -> None:
    result = make_result()
    assert result.findings == []
    assert result.charts == []
    assert result.data_quality_notes == []
    assert result.assumptions == []
    assert result.limitations == []


def test_result_default_lists_are_not_shared() -> None:
    a, b = make_result(), make_result()
    a.charts.append("x.png")
    assert b.charts == []


def test_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        make_result(confidence="certain")


def test_result_requires_confidence() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult(summary="s")  # type: ignore[call-arg]


def test_result_json_round_trip() -> None:
    result = make_result(
        findings=[make_finding()],
        charts=["outputs/revenue.png"],
        data_quality_notes=["3 null values in region"],
        assumptions=["Fiscal year equals calendar year"],
        limitations=["Only 2 quarters of data"],
    )
    assert AnalysisResult.model_validate_json(result.model_dump_json()) == result


# --- Contract snapshot: changing these is a breaking change for callers -------


def test_contract_field_names_are_stable() -> None:
    assert list(AnalysisRequest.model_fields) == ["question", "dataset_paths", "context", "max_steps"]
    assert list(Finding.model_fields) == ["statement", "evidence", "method", "code"]
    assert list(AnalysisResult.model_fields) == [
        "summary",
        "findings",
        "charts",
        "data_quality_notes",
        "assumptions",
        "limitations",
        "confidence",
    ]
