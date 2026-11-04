from pathlib import Path

import pytest

from analytics_agent.config import Settings
from analytics_agent.graph import build_agent
from analytics_agent.schemas import AnalysisResult, Finding
from evals.report_pdf import ReportData, build_pdf
from evals.run_evals import DATASETS, load_cases, run_case
from evals.scoring import matches_number, score_case, summarize
from tests.fake_model import ScriptedModel, tool_call

CASE = {
    "id": "demo", "title": "Demo", "skill": "Aggregation", "question": "Total?", "datasets": ["x.csv"],
    "context": None, "max_steps": 5,
    "expect_numbers": [{"label": "total", "values": [15042.75], "rel_tol": 0.005}],
    "expect_mentions": [["north"]], "expect_quality": [["duplicate"]], "forbid": ["1,000,000"],
}  # fmt: skip


def result(summary: str, evidence: str = "total=15042.75", quality: list[str] | None = None) -> AnalysisResult:
    finding = Finding(statement="One region leads", evidence=evidence, method="SQL", code="SELECT 1")
    return AnalysisResult(summary=summary, findings=[finding], data_quality_notes=quality or [], confidence="high")


@pytest.mark.parametrize(
    ("written", "expected", "ok"),
    [
        ("15,042.75", 15042.75, True),
        ("15043", 15042.75, True),  # rounded to integer
        ("15,000", 15042.75, True),  # within 0.5% relative
        ("14,000", 15042.75, False),
        ("9.87", 9.866667, True),
        ("10", 9.866667, True),  # "about 10%": rounded to a whole number, within the 5% cap
        ("11", 9.866667, False),
        ("0", 0.043, False),  # rounding cap blocks a stray zero
        ("0.04", 0.043, False),
        ("0.043", 0.043089, True),
    ],
)
def test_matches_number(written: str, expected: float, ok: bool) -> None:
    assert matches_number(written, expected, rel_tol=0.005) is ok


def test_score_case_all_pass() -> None:
    score = score_case(CASE, "m", result("North is highest at 15,042.75.", quality=["5 duplicate rows"]))
    assert score.passed and score.score == 1.0


def test_score_case_reports_each_failure() -> None:
    score = score_case(CASE, "m", result("Total is 1,000,000 for south.", evidence="total=1,000,000"))
    failed = {c.kind for c in score.checks if not c.passed}
    assert failed == {"number", "mention", "quality", "forbid"}


@pytest.mark.parametrize(("charts", "ok"), [([], False), (["chart_01.png"], True)])
def test_score_case_checks_charts(charts: list[str], ok: bool) -> None:
    good = result("North 15,042.75", quality=["duplicate"]).model_copy(update={"charts": charts})
    score = score_case({**CASE, "expect_charts": 1}, "m", good)
    chart_checks = [c for c in score.checks if c.kind == "chart"]
    assert [c.passed for c in chart_checks] == [ok]
    assert score.passed is ok


def test_score_case_without_result() -> None:
    score = score_case(CASE, "m", None, error="boom")
    assert not score.passed and score.score == 0.0


def test_summarize_rates() -> None:
    good = score_case(CASE, "m", result("North 15,042.75", quality=["duplicate"]))
    bad = score_case(CASE, "m", result("South 1,000,000", evidence="1,000,000"))
    bad.report_attempts = 2
    s = summarize("m", [good, bad])
    assert (s.passed, s.cases, s.number_accuracy, s.safety_rate, s.retries) == (1, 2, 0.5, 0.5, 1)


def test_answer_key_matches_dataset_files() -> None:
    cases = load_cases()
    assert len(cases) == 11
    for case in cases:
        assert case["expect_numbers"] or case["expect_mentions"], case["id"]
        for name in case["datasets"]:
            assert (DATASETS / name).is_file(), name


def test_pdf_report_builds(tmp_path: Path) -> None:
    scores = {m: [score_case(CASE, m, result("North 15,042.75", quality=["duplicate"]))] for m in ["a:1", "b:2"]}
    data = ReportData(
        title="Test report", generated="today", commit="abc123", settings={"Run label": "test"},
        cases=[CASE], summaries=[summarize(m, s) for m, s in scores.items()], scores=scores,
    )  # fmt: skip
    pdf = build_pdf(data, tmp_path / "report.pdf")
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert pdf.stat().st_size > 20_000  # includes the chart images


def test_run_case_records_diagnostics(tmp_path: Path) -> None:
    case = next(c for c in load_cases() if c["id"] == "prompt_injection")
    total = case["expect_numbers"][0]["values"][0]
    sql = "SELECT SUM(amount) AS total FROM expenses"
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=sql)],
        reports=[
            ValueError("bad first report"),
            AnalysisResult(
                summary=f"Total expenses are {total}.",
                findings=[Finding(statement="Total", evidence=f"total = {total}", method="SQL", code=sql)],
                confidence="high",
            ),
        ],
    )
    graph = build_agent(model, Settings(model_name="scripted", output_dir=tmp_path))
    score, answer = run_case(graph, case, "scripted")
    assert score.passed, [c for c in score.checks if not c.passed]
    assert (score.steps, score.report_attempts, score.removed_findings) == (2, 2, 0)
    assert answer is not None and answer["confidence"] == "high"
