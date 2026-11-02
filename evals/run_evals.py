"""Run the eval cases against one or more models and save a PDF + JSON report.

Usage:
    python -m evals.run_evals                         # all models in evals/models.json
    python -m evals.run_evals --models qwen3.5:9b     # a subset (names from models.json)
    python -m evals.run_evals --provider openai --models gpt-x   # a model not in the file
    python -m evals.run_evals --cases ab_test,prompt_injection

Each run writes evals/reports/<timestamp>/ (report.pdf, results.json, answers/) and appends
to evals/reports/history.csv.
"""

import argparse
import csv
import json
import logging
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import ValidationError

from analytics_agent.config import Settings
from analytics_agent.graph import build_agent, load_model
from analytics_agent.schemas import AnalysisRequest, AnalysisResult
from evals.report_pdf import ReportData, build_pdf
from evals.scoring import CaseScore, ModelSummary, score_case, summarize

HERE = Path(__file__).parent
DATASETS = HERE / "datasets"
REPORTS = HERE / "reports"
log = logging.getLogger("evals")


def load_cases(selected: list[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if selected:
        unknown = set(selected) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in selected]
    return cases


def load_models(names: list[str] | None, provider: str | None) -> list[dict[str, str]]:
    configured: list[dict[str, str]] = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    if not names:
        return configured
    by_name = {m["name"]: m for m in configured}
    return [by_name.get(n) or {"provider": provider or "", "name": n} for n in names]


def run_case(graph: Any, case: dict[str, Any], model: str) -> tuple[CaseScore, dict[str, Any] | None]:
    request = AnalysisRequest(
        question=case["question"],
        dataset_paths=[str(DATASETS / d) for d in case["datasets"]],
        context=case["context"],
        max_steps=case["max_steps"],
    )
    started = time.monotonic()
    state: dict[str, Any] = {}
    error = None
    with get_usage_metadata_callback() as usage:
        try:
            for state in graph.stream(request.model_dump(), stream_mode="values"):
                pass
        except Exception as exc:  # an eval run must survive one bad case
            error = f"{type(exc).__name__}: {exc}"
            log.exception("case %s failed for %s", case["id"], model)
    result = None
    if error is None:
        try:
            result = AnalysisResult.model_validate({k: state[k] for k in AnalysisResult.model_fields if k in state})
        except ValidationError as exc:
            error = f"invalid result: {exc}"
    score = score_case(case, model, result, error)
    score.seconds = time.monotonic() - started
    score.steps = int(state.get("steps", 0))
    score.tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
    score.report_attempts = int(state.get("report_attempts", 0))
    score.removed_findings = int(state.get("removed_findings", 0))
    return score, (result.model_dump(mode="json") if result else None)


def warm_up(settings: Settings) -> None:
    """Load the model into memory first so the first case's time is not inflated."""
    try:
        load_model(settings).invoke("Reply with OK.")
    except Exception as exc:
        log.warning("warm-up failed for %s: %s", settings.model_name, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="comma-separated model names (default: all in evals/models.json)")
    parser.add_argument("--provider", help="provider for models not listed in models.json")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--label", default="", help="optional note saved with the run, e.g. 'prompt v2'")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "analytics_agent.graph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cases = load_cases(args.cases.split(",") if args.cases else None)
    models = load_models(args.models.split(",") if args.models else None, args.provider)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = REPORTS / stamp
    (out_dir / "answers").mkdir(parents=True, exist_ok=True)
    base = Settings.from_env()
    base = replace(base, output_dir=HERE / ".runs" / stamp)

    all_scores: dict[str, list[CaseScore]] = {}
    summaries: list[ModelSummary] = []
    for spec in models:
        settings = replace(base, model_name=spec["name"], model_provider=spec["provider"] or None)
        log.info("=== %s (%d cases)", spec["name"], len(cases))
        warm_up(settings)
        graph = build_agent(settings=settings)
        scores = []
        for case in cases:
            score, answer = run_case(graph, case, spec["name"])
            scores.append(score)
            log.info("%-18s %-14s %-5s score=%.0f%% %.0fs", spec["name"], case["id"],
                     "PASS" if score.passed else "FAIL", score.score * 100, score.seconds)  # fmt: skip
            _write_json(out_dir / "answers" / _slug(spec["name"]) / f"{case['id']}.json",
                        {"score": score.to_dict(), "result": answer})  # fmt: skip
        all_scores[spec["name"]] = scores
        summaries.append(summarize(spec["name"], scores))

    commit = _git_commit()
    run_settings = {
        "Run label": args.label or "-",
        "Per-tool timeout": f"{base.tool_timeout_s:g}s",
        "Total run limit": f"{base.run_timeout_s:g}s",
        "Report format method": base.structured_output_method,
    }
    _write_json(out_dir / "results.json", {
        "generated": stamp, "commit": commit, "label": args.label, "settings": run_settings,
        "summaries": [asdict(s) for s in summaries],
        "scores": {m: [s.to_dict() for s in ss] for m, ss in all_scores.items()},
    })  # fmt: skip
    pdf = build_pdf(
        ReportData(
            title="Analytics Agent: Model Evaluation",
            generated=datetime.now().strftime("%d %B %Y, %H:%M"),
            commit=commit, settings=run_settings, cases=cases, summaries=summaries, scores=all_scores,
        ),
        out_dir / "report.pdf",
    )  # fmt: skip
    _append_history(stamp, commit, args.label, summaries)
    print(f"\nReport: {pdf.resolve()}")
    for s in summaries:
        print(f"  {s.model:<16} passed {s.passed}/{s.cases}  score {s.avg_score:.0%}  avg {s.avg_seconds:.0f}s")


def _append_history(stamp: str, commit: str, label: str, summaries: list[ModelSummary]) -> None:
    path = REPORTS / "history.csv"
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["run", "commit", "label", "model", "passed", "cases", "avg_score",
                             "number_accuracy", "quality_rate", "safety_rate", "removed_findings",
                             "retries", "errors", "avg_seconds", "avg_tokens"])  # fmt: skip
        for s in summaries:
            writer.writerow([stamp, commit, label, s.model, s.passed, s.cases, f"{s.avg_score:.3f}",
                             _opt(s.number_accuracy), _opt(s.quality_rate), _opt(s.safety_rate),
                             s.removed_findings, s.retries, s.errors, f"{s.avg_seconds:.1f}", f"{s.avg_tokens:.0f}"])  # fmt: skip


def _opt(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        # ignore untracked files: the run's own report folder must not count as a change
        status = ["git", "status", "--porcelain", "--untracked-files=no"]
        dirty = subprocess.run(status, capture_output=True, text=True, check=True)
        return sha.stdout.strip() + (" (with uncommitted changes)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in name)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
