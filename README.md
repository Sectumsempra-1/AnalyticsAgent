# Data Analytics Agent

A LangGraph agent that answers a natural-language question about tabular data
(CSV / Parquet / Excel) by writing and running code, and returns **structured,
evidence-backed findings**. It is built to run as a **subgraph** of a larger
multi-agent system: the researcher agent sends it a question plus data files and
gets an `AnalysisResult` back.

```
AnalysisRequest ─▶ prepare ─▶ agent ⇄ tools ─▶ finalize ─▶ verify ─▶ AnalysisResult
                   load files  model picks     model writes  drop findings whose
                   as tables   profile_dataset the report    numbers are not in
                               run_sql ...     (validated)   tool output
```

> **Security warning:** code written by the model runs in a *development* sandbox
> (a subprocess with Python-level guards). It is **not a security boundary**.
> Production use needs a container-level sandbox. See [Limitations](#limitations).

## Contents
- [Quick start](#quick-start)
- [The contract](#the-contract)
- [Using it as a subgraph](#using-it-as-a-subgraph)
- [Configuration](#configuration)
- [What the agent guarantees](#what-the-agent-guarantees)
- [Limitations](#limitations)
- [Evals](#evals)
- [Development](#development)

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[ollama]"        # or ".[openai]" for an OpenAI-compatible server
cp .env.example .env              # then edit .env if needed
```

The default model is a free local one, served by [Ollama](https://ollama.com):

```bash
ollama pull qwen3.5:9b            # ~6.6 GB; runs on an 8 GB GPU
```

### From the command line

```bash
analytics-agent ask "Which region has the highest total sales?" evals/datasets/sales_by_region.csv
analytics-agent ask "Compare Q1 and Q2 spending" budget.xlsx --context "amounts are in SAR"
analytics-agent ask "..." a.csv b.parquet --show-code      # also print the code behind each finding
analytics-agent ask "..." data.csv --json                  # raw AnalysisResult JSON
```

Progress (each tool call) is printed while the agent works, followed by the answer,
findings, data-quality notes, limitations and chart paths. Run `analytics-agent ask --help`
for every option. Exit code: `0` on success, `2` on bad input, configuration or model errors.

### From Python

```python
from analytics_agent import AnalysisRequest, AnalysisResult
from analytics_agent.graph import build_agent

agent = build_agent()  # model and limits come from env vars / .env
request = AnalysisRequest(
    question="Which region has the highest total sales amount, and what is that total?",
    dataset_paths=["evals/datasets/sales_by_region.csv"],
)
result = AnalysisResult.model_validate(agent.invoke(request.model_dump()))
print(result.summary)
```

A run takes roughly 20–90 seconds on a local GPU, depending on the question.

## The contract

Both models live in `src/analytics_agent/schemas.py` and reject unknown fields
(`extra="forbid"`), so contract drift fails loudly. Renaming or retyping a field is
a breaking change for callers.

### Input: `AnalysisRequest`

| Field | Type | Default | Notes |
|---|---|---|---|
| `question` | `str` | required | Non-empty. |
| `dataset_paths` | `list[str]` | required | At least one. `.csv`, `.parquet` or `.xlsx`. Existence is checked at run time. |
| `context` | `str \| None` | `None` | Domain notes, column meanings, units. Background only: it cannot override the agent's rules. |
| `max_steps` | `int` | `15` | 1–50. Upper bound on model turns. |

Each file becomes one SQL/pandas table named after the file (`Sales 2024.csv` →
`sales_2024`); each Excel sheet becomes its own table (`budget.xlsx`, sheet `Q1` →
`budget_q1`).

### Output: `AnalysisResult`

| Field | Type | Meaning |
|---|---|---|
| `summary` | `str` | A short, direct answer to the question. |
| `findings` | `list[Finding]` | Each has `statement`, `evidence` (exact figures from tool output), `method`, `code` (the SQL/Python that produced it). |
| `charts` | `list[str]` | Paths of PNG files the agent saved. |
| `data_quality_notes` | `list[str]` | Nulls, duplicates, outliers, type problems found. |
| `assumptions` | `list[str]` | |
| `limitations` | `list[str]` | Includes notes added by the agent itself (removed findings, step/time limit reached). |
| `confidence` | `"low" \| "medium" \| "high"` | Lowered automatically when findings are removed or nothing could be verified. |

Real example (from the eval run, `qwen3.5:9b`, shortened):

```json
{
  "summary": "North region has the highest total sales amount at 15,042.75",
  "findings": [
    {
      "statement": "The North region has the highest total sales among all regions.",
      "evidence": "north 15042.75",
      "method": "SQL aggregation with SUM() grouped by region and ordered by total_sales DESC",
      "code": "SELECT region, SUM(amount) as total_sales FROM sales_by_region WHERE region IS NOT NULL GROUP BY region ORDER BY total_sales DESC"
    }
  ],
  "charts": [],
  "data_quality_notes": [
    "12 null values in the region column (3.0% of rows)",
    "5 duplicate rows detected in the dataset"
  ],
  "assumptions": ["Assumed that NULL values in region column should be excluded from regional sales analysis"],
  "limitations": ["Duplicate rows were not removed before aggregation - this may slightly inflate total sales figures"],
  "confidence": "medium"
}
```

### How failures come back

| Situation | Behaviour |
|---|---|
| Malformed request (empty question, wrong file type, `max_steps` out of range) | Raises `pydantic.ValidationError`. This is a caller bug. |
| File missing or unreadable | Returns a normal `AnalysisResult` with `confidence="low"` and the reason in `limitations`. No exception. |
| Step or time limit reached | The agent stops, writes a report from what it has, and says so in `limitations`. |
| Model can't produce a valid report (after 3 attempts) | Returns `confidence="low"` with the reason in `limitations`. |
| A tool call fails (bad SQL, Python error, timeout) | The error text goes back to the model, which can fix it and retry. |

Callers should **check `confidence`** and read `limitations` before using a result.

## Using it as a subgraph

Build the agent **once** and reuse it. It holds no per-run state, and each run gets its
own workspace folder under `AGENT_OUTPUT_DIR`.

### Option A (recommended): call it from a node

Use this when the researcher's state has its own key names. You map fields in and out,
and nothing leaks into the parent state by accident.

<!-- tested: tests/test_readme.py -->
```python
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from analytics_agent import AnalysisRequest, AnalysisResult
from analytics_agent.graph import build_agent

analyst = build_agent()  # reads MODEL_* / AGENT_* settings from the environment


class ResearcherState(TypedDict, total=False):
    question: str
    files: list[str]
    analysis: dict[str, Any]


def run_analysis(state: ResearcherState) -> dict[str, Any]:
    request = AnalysisRequest(question=state["question"], dataset_paths=state["files"])
    result = AnalysisResult.model_validate(analyst.invoke(request.model_dump()))
    return {"analysis": result.model_dump()}


builder = StateGraph(ResearcherState)
builder.add_node("analyst", run_analysis)
builder.add_edge(START, "analyst")
builder.add_edge("analyst", END)
researcher = builder.compile()

state = researcher.invoke({"question": "What is the total sales amount?", "files": ["data/sales.csv"]})
print(state["analysis"]["summary"])
```

### Option B: add the compiled graph directly as a node

Use this when the parent state uses the same key names as the contract. The parent
state needs the input keys (`question`, `dataset_paths`, optionally `context` and
`max_steps`) and whichever output keys it wants back.

<!-- tested: tests/test_readme.py -->
```python
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from analytics_agent.graph import build_agent


class ResearcherState(TypedDict, total=False):
    question: str  # input keys shared with the analyst
    dataset_paths: list[str]
    summary: str  # output keys shared with the analyst
    findings: list[dict[str, Any]]
    data_quality_notes: list[str]
    limitations: list[str]
    confidence: str


builder = StateGraph(ResearcherState)
builder.add_node("analyst", build_agent())
builder.add_edge(START, "analyst")
builder.add_edge("analyst", END)
researcher = builder.compile()

state = researcher.invoke({"question": "What is the total sales amount?", "dataset_paths": ["data/sales.csv"]})
print(state["summary"], state["confidence"])
```

Both examples are run by `tests/test_readme.py` (with a scripted model), so they stay
correct.

Notes:
- **Paths:** `dataset_paths` are read by the process running the agent. Relative paths
  resolve against its working directory, so pass absolute paths between services.
- **Inspecting the input/output schemas:** `analyst.get_input_jsonschema()` and
  `analyst.get_output_jsonschema()` return exactly `AnalysisRequest` and `AnalysisResult`.
- **Recursion limit:** the compiled graph sets its own (`2 × 50 + 10`), enough for the
  maximum `max_steps`.

## Configuration

All settings come from environment variables (or `.env`, which is gitignored). Real
environment variables take precedence over `.env`. The template is `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | **required** | e.g. `qwen3.5:9b`, `gpt-4.1`, your company's model id. |
| `MODEL_PROVIDER` | (inferred) | Any [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) provider: `ollama`, `openai`, `azure_openai`, `anthropic`, ... Install its package too. |
| `MODEL_BASE_URL` | (provider default) | Custom endpoint: a company-hosted or OpenAI-compatible server, or a remote Ollama. |
| `MODEL_TEMPERATURE` | `0` | Keep at 0 for repeatable analysis. |
| `MODEL_STRUCTURED_OUTPUT_METHOD` | `function_calling` | How the final report is requested: `function_calling`, `json_schema` or `json_mode`. If two attempts fail, a final `json_schema` attempt is made automatically. |
| `AGENT_OUTPUT_DIR` | `outputs` | Where run workspaces (converted data, charts) are written. |
| `AGENT_RUN_TIMEOUT_S` | `600` | Total time limit for one run, including dataset loading. |
| `AGENT_TOOL_TIMEOUT_S` | `30` | Time limit for each code execution. |

API keys use the provider's own variable names (`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, ...).

### Switching to the company's model

No code changes are needed. For an OpenAI-compatible server:

```bash
pip install -e ".[openai]"
```

```ini
MODEL_PROVIDER=openai
MODEL_NAME=<company-model-id>
MODEL_BASE_URL=https://llm.example.internal/v1
OPENAI_API_KEY=<key>
```

The model must support **tool calling**. Then re-run the evals (below) to check the new
model before relying on it.

You can also pass settings or a ready-made model in code, e.g. for tests:
`build_agent(model=my_chat_model, settings=Settings(model_name="..."))`.

## What the agent guarantees

- **No invented numbers.** After the report is written, every number in each finding's
  `statement` and `evidence` is checked against the actual tool outputs (allowing for
  rounding and percent/fraction conversions). Findings that fail are **removed**, a note
  is added to `limitations`, and `confidence` is lowered. Unsupported numbers in the
  `summary` are flagged too.
- **No invented currencies.** A currency is accepted only if the data (values or column
  names, e.g. `amount_usd`), the question or the `context` mentions it. Otherwise a
  symbol such as `$` is removed from the summary and findings, with a note in
  `limitations`. A currency name or code (`USD`, `riyals`, ...) is flagged and
  `confidence` is lowered.
- **Reproducible.** Every finding includes the code that produced it.
- **Data quality first.** The agent is instructed to profile each table (nulls,
  duplicates, outliers, types) before concluding, and to report what it found.
- **Correlation is not causation; small samples are flagged.** This is enforced by the
  prompt, and tested by the eval cases.
- **Data is untrusted.** Instructions found inside the data are ignored (tested by the
  `prompt_injection` eval case). Long text cells are truncated before the model sees them.
- **The model never sees full datasets.** It sees profiles, at most 100 SQL result rows, and
  output capped at 10,000 characters.
- **Hard limits:** the step limit (`max_steps`), a per-execution timeout, a total run
  timeout, and a graph recursion limit.
- **Source files are never modified.** They are converted to read-only Parquet copies
  in the run's workspace. SQL runs on a locked-down DuckDB connection that can only read
  those copies.

## Limitations

**Sandbox: development only.** Model-written Python runs in a separate process with:
an isolated interpreter, a minimal environment (no secrets passed), a restricted working
directory, a timeout that kills the process tree, and capped output. Network access and
starting processes are blocked by **Python-level patches**, which determined code can
bypass (e.g. via `ctypes`). **For production, run the agent, or at least the code
execution, inside a container-level sandbox:** no network, CPU and memory limits,
read-only filesystem except the workspace, and a non-root user.

Also:
- **No memory limit** on executed code. A large allocation can exhaust the host's memory.
- **No cleanup** of `AGENT_OUTPUT_DIR`. Each run leaves a workspace folder there, so
  schedule cleanup in production.
- **Only currencies are checked, not other units.** An invented unit such as `kg` or `km`
  is not caught. The known currencies are USD, EUR, GBP,
  JPY, INR, SAR and AED; others are not checked. State units in `context` so the agent
  can use them.
- **Rounding tolerance.** A figure written as `15,000` matches a true value of `15,042.75`.
  The check catches invented numbers, not imprecise ones.
- **Data size:** files are loaded on one machine, so very large datasets need more
  memory or pre-aggregation.
- **File types:** CSV (UTF-8, falling back to Latin-1), Parquet and XLSX only. Columns
  with mixed types are converted to numbers where possible, otherwise stored as text,
  with a note in the profile.
- **Results vary by model.** See the eval scores below. Always check `confidence` and
  `limitations`.

## Evals

`evals/` holds 10 synthetic datasets with planted, known answers. Each one tests a
skill: aggregation, growth rates, A/B tests, a correlation trap, outliers, joins, Excel
sheets, prompt injection, small samples, and messy numbers.

```bash
pip install -e ".[evals]"
python -m evals.run_evals                                # all models in evals/models.json
python -m evals.run_evals --models qwen3.5:9b            # one model
python -m evals.run_evals --provider openai --models <company-model-id>
python -m evals.run_evals --cases ab_test,prompt_injection --label "prompt v2"
```

Each run writes `evals/reports/<timestamp>/`, containing `report.pdf` (a shareable
comparison with charts), `results.json` and every model answer. It also appends one row
per model to `evals/reports/history.csv`. Each case is scored on: a valid result, numeric
accuracy against the known answers, required mentions (e.g. "confound" in the
correlation trap), data-quality issues found, and forbidden content (e.g. the injected
number). Speed and token use are recorded too.

Latest results (`evals/reports/2026-11-02_1512`, local RTX 3060 Ti 8 GB):

| Model | Passed | Score | Avg time / case |
|---|---|---|---|
| **qwen3.5:9b** (default) | 10 / 10 | 100% | 34 s |
| gpt-oss:20b | 10 / 10 | 100% | 86 s |
| qwen3:8b | 7 / 10 | 90% | 50 s |
| granite4.2:8b (baseline run only) | 0 / 10 | 37% | 270 s |

Ten cases is a sanity check, not a benchmark. Re-run the evals whenever you change the
model or the prompts.

## Development

```bash
pip install -e ".[dev]"
pytest                    # unit + graph tests with a scripted model: fast, no network
pytest -m smoke -s        # one end-to-end run against the real model in .env
```

```
src/analytics_agent/
  schemas.py   # the contract: AnalysisRequest, AnalysisResult, Finding
  graph.py     # build_agent(): the LangGraph wrapper (the only LangGraph code)
  tools.py     # Workspace + profile_dataset, run_sql, run_python, make_chart
  sandbox.py   # subprocess execution: timeout, output caps, network/process blocks
  verify.py    # removes findings whose numbers are not in tool output
  prompts.py   # system and report prompts
  config.py    # Settings from env vars / .env
  cli.py       # the analytics-agent command
evals/         # datasets, cases.jsonl, run_evals.py, scoring, PDF report
tests/         # pytest suite; fake_model.py is the scripted model
```

Pinned: `langgraph==1.2.19`. LangGraph's API changes between versions, so check its
docs before upgrading the pin.
