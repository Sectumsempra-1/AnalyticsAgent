"""Mechanical check that reported numbers actually appeared in tool output.

A number in a finding counts as supported if some number in the tool output rounds to
it (at the precision written), possibly as a percentage (0.25 <-> 25%). Findings with
unsupported numbers, or with no numbers at all, are removed and listed in limitations.
Currencies the data never mentions (a model adding "$") are removed or flagged.
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


def verify_result(result: AnalysisResult, tool_outputs: Iterable[str], request_text: str = "") -> AnalysisResult:
    """Drop findings whose figures are not in tool output; record why; adjust confidence.

    `request_text` (the question and context) can legitimately name a currency."""
    outputs = list(tool_outputs)
    if not outputs:
        return result.model_copy(
            update={
                "findings": [],
                "confidence": "low",
                "limitations": [*result.limitations, "No tools were run, so no findings could be verified."],
            }
        )
    result, currency_fixes, currency_notes = check_currencies(result, "\n".join([*outputs, request_text]))
    known = sorted({float(n.replace(",", "")) for text in outputs for n in extract_numbers(text)})
    kept: list[Finding] = []
    notes: list[str] = list(currency_notes)
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
    limitations = [*result.limitations, *currency_fixes, *notes]
    return result.model_copy(update={"findings": kept, "confidence": confidence, "limitations": limitations})


def check_currencies(result: AnalysisResult, support: str) -> tuple[AnalysisResult, list[str], list[str]]:
    """Handle currencies in the summary and findings that `support` never mentions.

    A currency is supported if any of its symbols, codes or names appear in `support`
    (tool output, question, context), e.g. a column named amount_usd supports "$".
    Unsupported symbols next to a number are removed (returned as fixes: the text is now
    correct). Unsupported codes and names can't be removed cleanly from a sentence, so
    they are returned as problems, which lower confidence."""
    supported = {c.code for c in _CURRENCIES if c.mentioned_in(support)}
    unsupported = [c for c in _CURRENCIES if c.code not in supported]
    removed: set[str] = set()
    named: set[str] = set()

    def clean(text: str) -> str:
        for currency in unsupported:
            for symbol in currency.symbols:
                pattern = re.compile(rf"{re.escape(symbol)}\s?(?=[-\d.])|(?<=\d)\s?{re.escape(symbol)}")
                text, count = pattern.subn("", text)
                if count:
                    removed.add(symbol)
            if currency.name_pattern.search(text):
                named.add(currency.code)
        return text

    findings = [
        f.model_copy(update={"statement": clean(f.statement), "evidence": clean(f.evidence)})
        for f in result.findings
    ]
    result = result.model_copy(update={"summary": clean(result.summary), "findings": findings})
    fixes = [
        f"Removed currency symbol {s!r}: no currency appears in the data, question or context."
        for s in sorted(removed)
    ]
    problems = [
        f"The report names a currency ({code}) that does not appear in the data, question or context; "
        "treat the unit as unverified."
        for code in sorted(named)
    ]
    return result, fixes, problems


class _Currency:
    def __init__(self, code: str, symbols: Sequence[str], names: Sequence[str]) -> None:
        self.code = code
        self.symbols = tuple(symbols)
        words = "|".join(re.escape(w) for w in (code, *names))
        # letters must not touch the word, but underscores/digits may: amount_usd, usd2024
        self.name_pattern = re.compile(rf"(?<![A-Za-z])(?:{words})(?![A-Za-z])", re.IGNORECASE)

    def mentioned_in(self, text: str) -> bool:
        return any(s in text for s in self.symbols) or bool(self.name_pattern.search(text))


_CURRENCIES = (
    _Currency("USD", ["US$", "$"], ["dollar", "dollars", "US dollar"]),
    _Currency("EUR", ["€"], ["euro", "euros"]),
    _Currency("GBP", ["£"], ["sterling", "pound sterling"]),
    _Currency("JPY", ["¥"], ["yen"]),
    _Currency("INR", ["₹"], ["rupee", "rupees"]),
    _Currency("SAR", ["﷼"], ["riyal", "riyals", "rial", "rials"]),
    _Currency("AED", [], ["dirham", "dirhams"]),
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
