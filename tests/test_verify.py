import pytest

from analytics_agent.schemas import AnalysisResult, Finding
from analytics_agent.verify import extract_numbers, unsupported_numbers, verify_result

TOOL_OUTPUT = "total\n1395.5\n[1 rows]\nshare 0.2534 n=1,204 p=3.2e-05"


def finding(statement: str, evidence: str) -> Finding:
    return Finding(statement=statement, evidence=evidence, method="SQL", code="SELECT 1")


def result(*findings: Finding, summary: str = "Answer.", confidence: str = "high") -> AnalysisResult:
    return AnalysisResult(summary=summary, findings=list(findings), confidence=confidence)  # type: ignore[arg-type]


def test_extract_numbers() -> None:
    assert extract_numbers("n=1,204, mean 3.5, p=3e-05, list 1,2,3, .5") == ["1,204", "3.5", "3e-05", "1", "2", "3", ".5"]


@pytest.mark.parametrize(
    "text",
    [
        "total 1395.5",
        "total 1,395.5",
        "about 1396",  # rounded to integer
        "share 25.3%",  # 0.2534 as a percentage
        "n = 1204",
        "p = 3.2e-05",
    ],
)
def test_supported_numbers(text: str) -> None:
    known = [1395.5, 0.2534, 1204.0, 3.2e-05]
    assert unsupported_numbers(text, known) == []


@pytest.mark.parametrize("text", ["total 1400.0", "share 30%", "n = 1205"])
def test_unsupported_numbers(text: str) -> None:
    assert unsupported_numbers(text, [1395.5, 0.2534, 1204.0]) != []


def test_supported_finding_is_kept() -> None:
    out = verify_result(result(finding("Total is 1,395.5", "SUM(amount)=1395.5")), [TOOL_OUTPUT])
    assert len(out.findings) == 1
    assert out.confidence == "high"


def test_invented_number_removes_finding_and_lowers_confidence() -> None:
    good = finding("Total is 1395.5", "1395.5")
    bad = finding("Growth was 42%", "growth=42%")
    out = verify_result(result(good, bad), [TOOL_OUTPUT])
    assert out.findings == [good]
    assert out.confidence == "medium"
    assert any("Growth was 42%" in note and "42" in note for note in out.limitations)


def test_number_in_statement_is_checked_too() -> None:
    out = verify_result(result(finding("Revenue hit 9000", "SUM=1395.5")), [TOOL_OUTPUT])
    assert out.findings == []
    assert out.confidence == "low"  # every finding was removed


def test_finding_without_figures_is_removed() -> None:
    out = verify_result(result(finding("Sales look healthy", "the query showed growth")), [TOOL_OUTPUT])
    assert out.findings == []
    assert "no figures" in out.limitations[0]


def test_unsupported_summary_number_is_flagged() -> None:
    out = verify_result(result(finding("Total 1395.5", "1395.5"), summary="Total is 5000."), [TOOL_OUTPUT])
    assert len(out.findings) == 1
    assert out.confidence == "medium"
    assert "Summary contains figures" in out.limitations[0]


def test_no_tool_outputs_means_low_confidence() -> None:
    out = verify_result(result(finding("Total 1395.5", "1395.5")), [])
    assert out.findings == [] and out.confidence == "low"
