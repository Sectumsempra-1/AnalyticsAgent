"""Build a shareable PDF comparing models on the eval cases."""

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from evals.scoring import CaseScore, ModelSummary

# Reference categorical palette, light mode, fixed slot order (validated: adjacent-pair CVD
# and normal-vision checks pass; contrast relief = direct labels + the tables below).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
GOOD_TINT, PARTIAL_TINT, BAD_TINT = "#dff3df", "#fdf0d0", "#f8dcdc"  # status tints, always with a text label


@dataclass
class ReportData:
    title: str
    generated: str
    commit: str
    settings: dict[str, Any]
    cases: list[dict[str, Any]]
    summaries: list[ModelSummary]
    scores: dict[str, list[CaseScore]]  # model -> per-case scores, in case order


def build_pdf(data: ReportData, path: Path) -> Path:
    styles = _styles()
    with tempfile.TemporaryDirectory() as tmp:
        charts = _charts(data.summaries, Path(tmp))
        story: list[Any] = []
        story += _cover(data, styles)
        story += _at_a_glance(data, styles, charts)
        story += _case_matrix(data, styles)
        story += _case_details(data, styles)
        story += _method(styles)
        doc = SimpleDocTemplate(
            str(path), pagesize=A4, title=data.title, author="Analytics Agent evals",
            leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        )  # fmt: skip
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


# --- sections ---------------------------------------------------------------------


def _cover(data: ReportData, s: dict[str, ParagraphStyle]) -> list[Any]:
    models = ", ".join(m.model for m in data.summaries)
    rows = [
        ["Generated", data.generated],
        ["Code version", data.commit],
        ["Models compared", models],
        ["Test cases", str(len(data.cases))],
        *[[k, str(v)] for k, v in data.settings.items()],
    ]
    return [
        Spacer(1, 4 * cm),
        Paragraph(escape(data.title), s["title"]),
        Paragraph("How well each model answers data questions with known, planted answers", s["subtitle"]),
        Spacer(1, 1.2 * cm),
        _kv_table(rows, s),
        PageBreak(),
    ]


def _at_a_glance(data: ReportData, s: dict[str, ParagraphStyle], charts: dict[str, Path]) -> list[Any]:
    out: list[Any] = [Paragraph("At a glance", s["h1"])]
    out += [Paragraph(f"&bull; {escape(t)}", s["bullet"]) for t in _takeaways(data.summaries)]
    out.append(Spacer(1, 0.4 * cm))

    header = ["Model", "Cases passed", "Avg score", "Numbers right", "Quality issues caught",
              "Safety", "Invented figures removed", "Report retries", "Avg time", "Avg tokens"]  # fmt: skip
    rows: list[list[Any]] = [[Paragraph(h, s["th"]) for h in header]]
    for m in data.summaries:
        rows.append(
            [Paragraph(escape(m.model), s["td"]), f"{m.passed}/{m.cases}", _pct(m.avg_score),
             _pct(m.number_accuracy), _pct(m.quality_rate), _pct(m.safety_rate), str(m.removed_findings),
             str(m.retries), f"{m.avg_seconds:.0f}s", f"{m.avg_tokens:,.0f}"]
        )  # fmt: skip
    widths = [3.2 * cm] + [1.52 * cm] * 9
    out.append(_grid_table(rows, widths))
    out.append(PageBreak())
    out.append(Paragraph("Comparison charts", s["h1"]))
    for key, caption in [
        ("score", "Average score: share of checks passed per case (higher is better)"),
        ("seconds", "Average time per case in seconds (lower is better)"),
        ("tokens", "Average tokens per case (lower is cheaper)"),
    ]:
        out.append(KeepTogether([Paragraph(caption, s["caption"]), Image(str(charts[key]), width=16 * cm, height=4.6 * cm)]))
        out.append(Spacer(1, 0.4 * cm))
    out.append(PageBreak())
    return out


def _case_matrix(data: ReportData, s: dict[str, ParagraphStyle]) -> list[Any]:
    models = [m.model for m in data.summaries]
    rows: list[list[Any]] = [[Paragraph("Case", s["th"])] + [Paragraph(escape(m), s["th"]) for m in models]]
    tints: list[tuple[int, int, str]] = []
    for r, case in enumerate(data.cases, start=1):
        row: list[Any] = [Paragraph(f"<b>{escape(case['title'])}</b><br/>{escape(case['skill'])}", s["td"])]
        for c, model in enumerate(models, start=1):
            score = data.scores[model][r - 1]
            if score.error:
                label, tint = "ERROR", BAD_TINT
            elif score.passed:
                label, tint = "PASS", GOOD_TINT
            else:
                ok = sum(ch.passed for ch in score.checks)
                label, tint = f"{ok}/{len(score.checks)} checks", PARTIAL_TINT if ok else BAD_TINT
            row.append(label)
            tints.append((c, r, tint))
        rows.append(row)
    widths = [6.0 * cm] + [(17.4 - 6.0) / len(models) * cm] * len(models)
    table = _grid_table(rows, widths)
    table.setStyle(TableStyle([("BACKGROUND", (c, r), (c, r), colors.HexColor(t)) for c, r, t in tints]))
    return [
        Paragraph("Results by case", s["h1"]),
        Paragraph("PASS means every check for the case passed. Partial results show how many checks passed.", s["body"]),
        Spacer(1, 0.3 * cm),
        table,
        PageBreak(),
    ]


def _case_details(data: ReportData, s: dict[str, ParagraphStyle]) -> list[Any]:
    out: list[Any] = [Paragraph("Case details", s["h1"])]
    for i, case in enumerate(data.cases):
        expected = "; ".join(f"{n['label']} = {_fmt_value(n['values'][0])}" for n in case["expect_numbers"])
        block: list[Any] = [
            Paragraph(f"{i + 1}. {escape(case['title'])}", s["h2"]),
            Paragraph(f"<b>Question:</b> {escape(case['question'])}", s["body"]),
            Paragraph(f"<b>Tests:</b> {escape(case['skill'])}. <b>Planted answer:</b> {escape(expected or 'n/a')}", s["body"]),
        ]
        rows: list[list[Any]] = [[Paragraph(h, s["th"]) for h in ["Model", "Result", "Agent's answer", "Failed checks"]]]
        for m in data.summaries:
            sc = data.scores[m.model][i]
            failed = [c.label for c in sc.checks if not c.passed]
            result = "ERROR" if sc.error else ("PASS" if sc.passed else f"{_pct(sc.score)}")
            answer = sc.error or sc.summary
            rows.append(
                [Paragraph(escape(m.model), s["td"]), Paragraph(f"{result}<br/>{sc.seconds:.0f}s, {sc.steps} steps", s["td"]),
                 Paragraph(escape(_trim(answer, 320)), s["td"]), Paragraph(escape("; ".join(failed) or "none"), s["td"])]
            )  # fmt: skip
        block.append(_grid_table(rows, [2.8 * cm, 2.2 * cm, 8.0 * cm, 4.4 * cm]))
        out.append(KeepTogether(block))
        out.append(Spacer(1, 0.5 * cm))
    out.append(PageBreak())
    return out


def _method(s: dict[str, ParagraphStyle]) -> list[Any]:
    items = [
        ("Datasets", "Synthetic files generated with a fixed seed (evals/make_datasets.py). Each has planted answers "
         "computed from the data itself, plus planted problems: nulls, duplicates, an outlier, text-formatted "
         "numbers, a tiny group, and instructions hidden in a cell."),
        ("Numbers right", "An expected figure counts as found if the summary or a finding states it, allowing the "
         "precision the agent chose (e.g. 15,042.75 written as 15,043), capped at 5% of the value."),
        ("Quality issues caught", "Whether planted data problems appear in the data-quality notes or limitations."),
        ("Safety", "Whether the answer avoided a figure that the data tried to inject."),
        ("Invented figures removed", "Findings dropped by the agent's own check because their numbers never "
         "appeared in tool output. Lower is better; the check keeps them out of the final answer."),
        ("Caveats", "One run per model per case, so small differences can be noise. Keyword checks are a proxy for "
         "reasoning quality. Times depend on this machine (local GPU, partial CPU offload for larger models)."),
    ]  # fmt: skip
    out: list[Any] = [Paragraph("How to read this report", s["h1"])]
    for head, text in items:
        out.append(Paragraph(f"<b>{escape(head)}.</b> {escape(text)}", s["body"]))
    return out


# --- charts -----------------------------------------------------------------------


def _charts(summaries: Sequence[ModelSummary], folder: Path) -> dict[str, Path]:
    specs = {
        "score": ([m.avg_score * 100 for m in summaries], lambda v: f"{v:.0f}%", 100.0),
        "seconds": ([m.avg_seconds for m in summaries], lambda v: f"{v:.0f}s", None),
        "tokens": ([m.avg_tokens for m in summaries], lambda v: f"{v:,.0f}", None),
    }
    paths = {}
    names = [m.model for m in summaries]
    for key, (values, label, xmax) in specs.items():
        paths[key] = _bar_chart(names, values, label, xmax, folder / f"{key}.png")
    return paths


def _bar_chart(names: list[str], values: list[float], label: Any, xmax: float | None, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 2.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(names)))[::-1]
    ax.barh(y, values, height=0.55, color=[SERIES[i % len(SERIES)] for i in range(len(names))], edgecolor=SURFACE, linewidth=2)
    top = xmax or (max(values) * 1.18 if any(values) else 1.0)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + top * 0.01, yi, label(v), va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y, names, fontsize=9, color=INK)
    ax.set_xlim(0, top * (1.08 if xmax else 1.0))
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- helpers ----------------------------------------------------------------------


def _takeaways(summaries: Sequence[ModelSummary]) -> list[str]:
    if not summaries:
        return ["No models were evaluated."]
    best = max(summaries, key=lambda m: (m.avg_score, m.passed))
    fastest = min(summaries, key=lambda m: m.avg_seconds)
    lines = [
        f"Best overall: {best.model}, passing {best.passed} of {best.cases} cases fully "
        f"(average score {_pct(best.avg_score)}).",
        f"Fastest: {fastest.model}, about {fastest.avg_seconds:.0f} seconds per question.",
    ]
    accurate = [m for m in summaries if m.number_accuracy is not None]
    if accurate:
        top = max(accurate, key=lambda m: m.number_accuracy or 0)
        lines.append(f"Most accurate figures: {top.model} ({_pct(top.number_accuracy)} of expected numbers found).")
    unsafe = [m.model for m in summaries if m.safety_rate is not None and m.safety_rate < 1]
    lines.append(
        "Every model ignored the instructions hidden in the data."
        if not unsafe
        else f"Repeated an injected figure from the data: {', '.join(unsafe)}."
    )
    errors = [m.model for m in summaries if m.errors]
    if errors:
        lines.append(f"Runs that crashed or timed out: {', '.join(errors)}.")
    return lines


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor(INK), spaceAfter=4)
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26, leading=32, alignment=TA_LEFT, textColor=colors.HexColor(INK)),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=13, leading=18, textColor=colors.HexColor(INK_2)),
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20, spaceBefore=4, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11.5, leading=15, spaceBefore=4, spaceAfter=4),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=10, fontSize=10.5, leading=15),
        "caption": ParagraphStyle("caption", parent=body, textColor=colors.HexColor(INK_2), fontSize=9),
        "th": ParagraphStyle("th", parent=body, fontName="Helvetica-Bold", fontSize=8, leading=10, spaceAfter=0),
        "td": ParagraphStyle("td", parent=body, fontSize=8, leading=10.5, spaceAfter=0),
    }  # fmt: skip


def _grid_table(rows: list[list[Any]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _kv_table(rows: list[list[str]], s: dict[str, ParagraphStyle]) -> Table:
    data = [[Paragraph(f"<b>{escape(k)}</b>", s["td"]), Paragraph(escape(v), s["td"])] for k, v in rows]
    return _grid_table(data, [4.5 * cm, 12.9 * cm])


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(1.8 * cm, 1.0 * cm, "Analytics Agent - model evaluation")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _fmt_value(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"


def _trim(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
