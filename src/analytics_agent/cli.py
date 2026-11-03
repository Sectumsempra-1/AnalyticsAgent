"""Command line: ask the agent a question about one or more data files.

    analytics-agent ask "Which region sold the most?" data/sales.csv
    analytics-agent ask "Compare Q1 and Q2" budget.xlsx --context "amounts are in SAR"
    analytics-agent ask "..." data.csv --json          # raw AnalysisResult JSON
"""

import argparse
import json
import logging
import sys
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from analytics_agent.config import ConfigError, Settings
from analytics_agent.graph import build_agent
from analytics_agent.schemas import AnalysisRequest, AnalysisResult

WIDTH = 100


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles can't print every character
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s")
    try:
        request = AnalysisRequest(
            question=args.question,
            dataset_paths=[str(Path(p).resolve()) for p in args.files],
            context=args.context,
            max_steps=args.max_steps,
        )
        settings = Settings.from_env()
    except ValidationError as exc:
        return _fail(f"invalid request: {_first_error(exc)}")
    except ConfigError as exc:
        return _fail(f"configuration error: {exc}")

    progress = sys.stderr
    if not args.json:
        print(f"Analyzing with {settings.model_name} (usually 20-90 seconds)...", file=progress)
    started = time.monotonic()
    try:
        result = _run(request, settings, progress if not args.json else None)
    except KeyboardInterrupt:
        return _fail("stopped by user", code=130)
    except Exception as exc:  # e.g. model server not running
        return _fail(f"the agent failed: {type(exc).__name__}: {exc}")

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(f"Done in {time.monotonic() - started:.0f}s.\n", file=progress)
        print(render(result, show_code=args.show_code))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analytics-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask a question about data files")
    ask.add_argument("question", help="the question, in quotes")
    ask.add_argument("files", nargs="+", help="one or more .csv, .xlsx or .parquet files")
    ask.add_argument("--context", help="notes about the data: column meanings, units, ...")
    ask.add_argument("--max-steps", type=int, default=15, help="limit on analysis steps (1-50, default 15)")
    ask.add_argument("--show-code", action="store_true", help="print the code behind each finding")
    ask.add_argument("--json", action="store_true", help="print the raw result as JSON")
    ask.add_argument("-v", "--verbose", action="store_true", help="show detailed logs")
    return parser


def _run(request: AnalysisRequest, settings: Settings, progress: TextIO | None) -> AnalysisResult:
    """Run the agent, printing each tool call as it happens."""
    state: dict[str, Any] = {}
    for mode, chunk in build_agent(settings=settings).stream(request.model_dump(), stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
        elif progress is not None:
            for line in _progress_lines(chunk):
                print(line, file=progress, flush=True)
    return AnalysisResult.model_validate({k: state[k] for k in AnalysisResult.model_fields if k in state})


def _progress_lines(update: dict[str, Any]) -> list[str]:
    lines = []
    for node, data in update.items():
        if node == "agent" and data:
            for message in data.get("messages", []):
                for call in getattr(message, "tool_calls", []):
                    detail = " ".join(str(v) for v in call["args"].values()).replace("\n", " ")
                    lines.append(_clip(f"  - {call['name']}: {detail}", WIDTH))
        elif node == "finalize":
            lines.append("  - writing the report")
        elif node == "verify":
            lines.append("  - checking every number against the tool output")
    return lines


def render(result: AnalysisResult, show_code: bool = False) -> str:
    """Format a result for reading in a terminal."""
    out = [_heading("Answer"), _wrap(result.summary), "", f"Confidence: {result.confidence.upper()}"]
    if result.findings:
        out += ["", _heading("Findings")]
        for i, f in enumerate(result.findings, start=1):
            out += [_wrap(f.statement, f"{i}. "), _wrap(f.evidence, "   Evidence: "), _wrap(f.method, "   Method:   ")]
            if show_code:
                out += ["   Code:", textwrap.indent(f.code.strip(), "     ")]
            out.append("")
    sections = [
        ("Data quality", result.data_quality_notes),
        ("Assumptions", result.assumptions),
        ("Limitations", result.limitations),
    ]
    for title, items in sections:
        if items:
            out += [_heading(title), *(_wrap(item, "- ") for item in items), ""]
    if result.charts:  # paths are never wrapped, so they can be copied
        out += [_heading("Charts"), *(f"- {Path(c).resolve()}" for c in result.charts), ""]
    return "\n".join(out).rstrip() + "\n"


def _heading(title: str) -> str:
    return f"{title}\n{'-' * len(title)}"


def _wrap(text: str, prefix: str = "") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=prefix, subsequent_indent=" " * len(prefix))


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    cause = error.get("ctx", {}).get("error")
    message = str(cause) if cause is not None else error["msg"]  # drop pydantic's "Value error, " prefix
    field = str(error["loc"][0]) if error["loc"] else ""
    return f"{field}: {message}" if field else message


def _fail(message: str, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
