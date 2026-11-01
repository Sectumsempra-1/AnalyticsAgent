"""Mechanical check that reported numbers actually appeared in tool output.

A number in a finding counts as supported if some number in the tool output rounds to
it (at the precision written), possibly as a percentage (0.25 <-> 25%). Findings with
unsupported numbers, or with no numbers at all, are removed and listed in limitations.
"""

import re
from collections.abc import Iterable, Sequence

from analytics_agent.schemas import AnalysisResult, Confidence, Finding

_NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+")
_LOWER: dict[Confidence, Confidence] = {"high": "medium", "medium": "low", "low": "low"}


def extract_numbers(text: str) -> list[str]:
    """Numbers as written (without sign; commas kept), e.g. '1,204', '0.25', '3e-05'."""
    return _NUMBER.findall(text)


def unsupported_numbers(text: str, known: Sequence[float]) -> list[str]:
    """Numbers in `text` that do not match any value in `known`."""
    return [n for n in extract_numbers(text) if not _is_supported(n, known)]


def verify_result(result: AnalysisResult, tool_outputs: Iterable[str]) -> AnalysisResult:
    """Drop findings whose figures are not in tool output; record why; adjust confidence."""
    outputs = list(tool_outputs)
    if not outputs:
        return result.model_copy(
            update={
                "findings": [],
                "confidence": "low",
                "limitations": [*result.limitations, "No tools were run, so no findings could be verified."],
            }
        )
    known = sorted({float(n.replace(",", "")) for text in outputs for n in extract_numbers(text)})
    kept: list[Finding] = []
    notes: list[str] = []
    for finding in result.findings:
        problem = _finding_problem(finding, known)
        if problem:
            notes.append(f"Removed finding {finding.statement!r}: {problem}.")
        else:
            kept.append(finding)
    bad_summary = unsupported_numbers(result.summary, known)
    if bad_summary:
        notes.append(f"Summary contains figures not found in tool output: {', '.join(bad_summary)}.")

    confidence = result.confidence
    if notes:
        confidence = _LOWER[confidence]
    if result.findings and not kept:
        confidence = "low"
    return result.model_copy(
        update={"findings": kept, "confidence": confidence, "limitations": [*result.limitations, *notes]}
    )


def _finding_problem(finding: Finding, known: Sequence[float]) -> str | None:
    if not extract_numbers(finding.evidence):
        return "its evidence contains no figures"
    missing = unsupported_numbers(f"{finding.statement} {finding.evidence}", known)
    if missing:
        return f"figures not found in tool output ({', '.join(missing)})"
    return None


def _is_supported(written: str, known: Sequence[float]) -> bool:
    """True if some known value, or that value as a percent/fraction, rounds to `written`."""
    value = float(written.replace(",", ""))
    tolerance = _rounding_tolerance(written, value)
    return any(abs(c - value) <= tolerance for k in known for c in (abs(k), abs(k) * 100, abs(k) / 100))


def _rounding_tolerance(written: str, value: float) -> float:
    """Half a unit in the last written digit, e.g. '12.3' -> 0.05; scientific: 0.5% relative."""
    if "e" in written.lower():
        return abs(value) * 0.005
    decimals = len(written.split(".")[1]) if "." in written else 0
    return 0.5 * 10**-decimals + 1e-9
