"""Input/output contract for the analytics agent.

These models are the stable interface used by the calling agent (the researcher).
Changing a field name or type here is a breaking change for callers.
"""

from pathlib import PurePath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".parquet", ".xlsx"})
MAX_STEPS_LIMIT = 50

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Confidence = Literal["low", "medium", "high"]


class _Contract(BaseModel):
    """Base config: reject unknown fields so contract drift fails loudly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Finding(_Contract):
    """One evidence-backed conclusion drawn from executed code."""

    statement: NonEmptyStr = Field(description="The conclusion in plain language.")
    evidence: NonEmptyStr = Field(
        description="Exact figures from tool output that support the statement."
    )
    method: NonEmptyStr = Field(description="How the result was obtained (e.g. SQL aggregation, t-test).")
    code: NonEmptyStr = Field(description="The executed code that produced the evidence.")


class AnalysisRequest(_Contract):
    """What the caller sends: a question plus the datasets to analyze."""

    question: NonEmptyStr = Field(description="Natural-language question to answer.")
    dataset_paths: list[NonEmptyStr] = Field(
        min_length=1, description="Paths to CSV, Parquet, or XLSX files."
    )
    context: str | None = Field(
        default=None, description="Optional domain notes or column meanings."
    )
    max_steps: int = Field(
        default=15, ge=1, le=MAX_STEPS_LIMIT, description="Upper bound on agent tool steps."
    )

    @field_validator("dataset_paths")
    @classmethod
    def _check_extensions(cls, paths: list[str]) -> list[str]:
        """Reject unsupported file types. File existence is checked at run time."""
        for path in paths:
            suffix = PurePath(path).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise ValueError(f"unsupported file type {suffix or '(none)'!r} for {path!r}; allowed: {allowed}")
        return paths


class AnalysisResult(_Contract):
    """What the agent returns to the caller."""

    summary: NonEmptyStr = Field(description="Short, direct answer to the question.")
    findings: list[Finding] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list, description="Paths to saved chart images.")
    data_quality_notes: list[str] = Field(
        default_factory=list, description="Nulls, duplicates, outliers, type problems found."
    )
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: Confidence
