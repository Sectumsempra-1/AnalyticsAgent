"""Score one agent result against a case's planted answers. Pure functions, no model calls."""

from dataclasses import asdict, dataclass, field
from typing import Any

from analytics_agent.schemas import AnalysisResult
from analytics_agent.verify import extract_numbers

ROUNDING_CAP = 0.05  # a written number may be rounded, but never by more than 5% of the value


@dataclass
class Check:
    kind: str  # number | mention | quality | forbid | valid
    label: str
    passed: bool


@dataclass
class CaseScore:
    case_id: str
    model: str
    checks: list[Check] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0
    steps: int = 0
    tokens: int = 0
    report_attempts: int = 0
    removed_findings: int = 0
    confidence: str | None = None
    summary: str = ""

    @property
    def score(self) -> float:
        return sum(c.passed for c in self.checks) / len(self.checks) if self.checks else 0.0

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)

    def rate(self, kind: str) -> float | None:
        relevant = [c for c in self.checks if c.kind == kind]
        return sum(c.passed for c in relevant) / len(relevant) if relevant else None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "score": self.score, "passed": self.passed}


def matches_number(written: str, expected: float, rel_tol: float) -> bool:
    """True if `written` equals `expected`, allowing rel_tol and rounding to the written precision."""
    value = float(written.replace(",", ""))
    if "e" in written.lower():
        rounding = abs(value) * 0.005
    else:
        decimals = len(written.split(".")[1]) if "." in written else 0
        rounding = 0.5 * 10**-decimals
    tolerance = max(rel_tol * abs(expected), min(rounding, ROUNDING_CAP * abs(expected)), 1e-9)
    return abs(value - abs(expected)) <= tolerance


def answer_text(result: AnalysisResult) -> str:
    """The parts of a result that state the answer: summary and findings."""
    parts = [result.summary] + [f"{f.statement} {f.evidence}" for f in result.findings]
    return "\n".join(parts)


def full_text(result: AnalysisResult) -> str:
    """Everything the result says, including notes and limitations."""
    return "\n".join(
        [answer_text(result), *result.data_quality_notes, *result.assumptions, *result.limitations]
    )


def score_case(case: dict[str, Any], model: str, result: AnalysisResult | None, error: str | None = None) -> CaseScore:
    score = CaseScore(case_id=case["id"], model=model, error=error)
    if result is None:
        score.checks.append(Check("valid", "valid AnalysisResult returned", False))
        return score
    score.confidence = result.confidence
    score.summary = result.summary
    answer, everything = answer_text(result), full_text(result).lower()
    quality_text = "\n".join([*result.data_quality_notes, *result.limitations]).lower()
    written = extract_numbers(answer)

    score.checks.append(Check("valid", "valid AnalysisResult returned", True))
    for exp in case["expect_numbers"]:
        found = any(matches_number(w, v, exp["rel_tol"]) for w in written for v in exp["values"])
        score.checks.append(Check("number", exp["label"], found))
    for options in case["expect_mentions"]:
        score.checks.append(Check("mention", " / ".join(options), any(o.lower() in everything for o in options)))
    for options in case["expect_quality"]:
        score.checks.append(Check("quality", " / ".join(options), any(o.lower() in quality_text for o in options)))
    for banned in case["forbid"]:
        score.checks.append(Check("forbid", f"does not state {banned!r}", banned.lower() not in answer.lower()))
    return score


@dataclass
class ModelSummary:
    model: str
    cases: int
    passed: int
    avg_score: float
    number_accuracy: float | None
    quality_rate: float | None
    safety_rate: float | None
    removed_findings: int
    retries: int
    errors: int
    avg_seconds: float
    avg_tokens: float
    avg_steps: float


def summarize(model: str, scores: list[CaseScore]) -> ModelSummary:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def kind_rate(kind: str) -> float | None:
        checks = [c for s in scores for c in s.checks if c.kind == kind]
        return sum(c.passed for c in checks) / len(checks) if checks else None

    return ModelSummary(
        model=model,
        cases=len(scores),
        passed=sum(s.passed for s in scores),
        avg_score=mean([s.score for s in scores]),
        number_accuracy=kind_rate("number"),
        quality_rate=kind_rate("quality"),
        safety_rate=kind_rate("forbid"),
        removed_findings=sum(s.removed_findings for s in scores),
        retries=sum(max(s.report_attempts - 1, 0) for s in scores),
        errors=sum(s.error is not None for s in scores),
        avg_seconds=mean([s.seconds for s in scores]),
        avg_tokens=mean([float(s.tokens) for s in scores]),
        avg_steps=mean([float(s.steps) for s in scores]),
    )
