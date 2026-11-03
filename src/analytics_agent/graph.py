"""LangGraph wrapper: build_agent() returns a compiled graph AnalysisRequest -> AnalysisResult.

Flow: prepare -> agent <-> tools -> finalize -> verify.
"""

import logging
import operator
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from analytics_agent.config import Settings
from analytics_agent.prompts import FINAL_REPORT_PROMPT, NO_TOOL_CALL_PROMPT, RETRY_PROMPT, build_system_prompt
from analytics_agent.schemas import MAX_STEPS_LIMIT, AnalysisRequest, AnalysisResult
from analytics_agent.tools import DatasetLoadError, Workspace, make_tools
from analytics_agent.verify import verify_result

log = logging.getLogger(__name__)

FINAL_REPORT_ATTEMPTS = 2  # with the configured method; then one STRICT_METHOD attempt
# Schema-constrained decoding (Ollama `format`, OpenAI response_format): slower but the
# output is forced into the schema's shape. Used only as the last resort.
STRICT_METHOD = "json_schema"
# Each step is an agent superstep plus a tools superstep; the rest covers fixed nodes.
RECURSION_LIMIT = 2 * MAX_STEPS_LIMIT + 10


class AgentState(TypedDict, total=False):
    # input (AnalysisRequest)
    question: str
    dataset_paths: list[str]
    context: str | None
    max_steps: int
    # output (AnalysisResult)
    summary: str
    findings: list[dict[str, Any]]
    charts: list[str]
    data_quality_notes: list[str]
    assumptions: list[str]
    limitations: list[str]
    confidence: str
    # internal
    messages: Annotated[list[AnyMessage], add_messages]
    tool_outputs: Annotated[list[str], operator.add]
    workspace_root: str
    steps: int
    deadline: float
    stop_reason: str
    done: bool
    report_attempts: int  # diagnostics for evals; not part of the output contract
    removed_findings: int


def load_model(settings: Settings) -> BaseChatModel:
    """Create the chat model named in settings (any provider supported by init_chat_model)."""
    kwargs: dict[str, Any] = {"temperature": settings.model_temperature}
    if settings.model_base_url:
        kwargs["base_url"] = settings.model_base_url
    if settings.model_provider == "ollama":
        # Ollama silently drops the start of the conversation beyond num_ctx
        kwargs["num_ctx"] = settings.context_tokens
    model = init_chat_model(settings.model_name, model_provider=settings.model_provider, **kwargs)
    assert isinstance(model, BaseChatModel)
    return model


def build_agent(
    model: BaseChatModel | None = None, settings: Settings | None = None
) -> CompiledStateGraph:
    """Compile the analytics agent graph.

    Invoke with an AnalysisRequest as a dict; the result is a dict matching AnalysisResult:
        out = build_agent().invoke(request.model_dump())
        result = AnalysisResult.model_validate(out)
    """
    settings = settings or Settings.from_env()
    model = model or load_model(settings)
    nodes = _Nodes(model, settings)

    builder = StateGraph(AgentState, input_schema=AnalysisRequest, output_schema=AnalysisResult)
    builder.add_node("prepare", nodes.prepare)
    builder.add_node("agent", nodes.agent)
    builder.add_node("tools", nodes.tools)
    builder.add_node("finalize", nodes.finalize)
    builder.add_node("verify", nodes.verify)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _after_prepare, ["agent", END])
    builder.add_conditional_edges("agent", _after_agent, ["tools", "finalize"])
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", "verify")
    builder.add_edge("verify", END)
    return builder.compile(name="analytics_agent").with_config(recursion_limit=RECURSION_LIMIT)


def _after_prepare(state: AgentState) -> Literal["agent", "__end__"]:
    return END if state.get("done") else "agent"


def _after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    if state.get("stop_reason"):
        return "finalize"
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalize"


class _Nodes:
    """Node functions, sharing the model and settings."""

    def __init__(self, model: BaseChatModel, settings: Settings) -> None:
        self.model = model
        self.settings = settings

    def prepare(self, state: AgentState) -> dict[str, Any]:
        started = time.time()  # the run time limit includes dataset loading
        # LangGraph does not validate input schemas or apply their defaults, so do it here.
        # A malformed request is a caller bug and raises ValidationError.
        request = AnalysisRequest.model_validate({k: state[k] for k in AnalysisRequest.model_fields if k in state})
        root = self.settings.output_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        try:
            workspace = Workspace.create(request.dataset_paths, root, timeout_s=self.settings.tool_timeout_s)
        except (FileNotFoundError, ValueError, DatasetLoadError) as exc:
            log.warning("dataset load failed: %s", exc)
            failure = AnalysisResult(
                summary="The analysis could not run because the datasets could not be loaded.",
                limitations=[f"Dataset loading failed: {exc}"],
                confidence="low",
            )
            return {**failure.model_dump(mode="json"), "done": True}
        system = build_system_prompt(workspace.describe_tables(), request.context, request.max_steps)
        return {
            "max_steps": request.max_steps,
            "workspace_root": str(workspace.root),
            "steps": 0,
            "deadline": started + self.settings.run_timeout_s,
            "messages": [SystemMessage(system), HumanMessage(request.question)],
        }

    def agent(self, state: AgentState) -> dict[str, Any]:
        steps, max_steps = state["steps"], state["max_steps"]
        if steps >= max_steps:
            return {"stop_reason": f"Stopped at the step limit ({max_steps}); the analysis may be incomplete."}
        if time.time() >= state["deadline"]:
            limit = self.settings.run_timeout_s
            return {"stop_reason": f"Stopped at the total time limit ({limit:g}s); the analysis may be incomplete."}
        tools = make_tools(self._workspace(state))
        reply = self.model.bind_tools(tools).invoke(state["messages"])
        log.info("step %d/%d: %d tool call(s)", steps + 1, max_steps, len(getattr(reply, "tool_calls", [])))
        return {"messages": [reply], "steps": steps + 1}

    def tools(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        available = {t.__name__: t for t in make_tools(self._workspace(state))}
        messages, outputs = [], []
        for call in last.tool_calls:
            output = _call_tool(available, call["name"], call["args"])
            messages.append(ToolMessage(content=output, tool_call_id=call["id"], name=call["name"]))
            outputs.append(output)
        return {"messages": messages, "tool_outputs": outputs}

    def finalize(self, state: AgentState) -> dict[str, Any]:
        """Ask for the structured report: retries with feedback, then one strict-format attempt."""
        base = [*state["messages"], HumanMessage(FINAL_REPORT_PROMPT)]
        messages = list(base)
        configured = self.settings.structured_output_method
        methods = [configured] * FINAL_REPORT_ATTEMPTS
        if configured != STRICT_METHOD:
            methods.append(STRICT_METHOD)
        error = "no attempt made"
        for attempt, method in enumerate(methods, start=1):
            if method != configured:
                messages = list(base)  # fresh, shorter context for the strict attempt
            result, error, retry = self._report_attempt(messages, method)
            if result is not None:
                return {**result.model_dump(mode="json"), "report_attempts": attempt}
            log.warning("final report attempt %d (%s) failed: %s", attempt, method, error)
            messages.append(HumanMessage(retry))
        fallback = AnalysisResult(
            summary="The agent ran its analysis but could not produce a valid structured report.",
            limitations=[f"Final report failed validation: {error}"],
            confidence="low",
        )
        return {**fallback.model_dump(mode="json"), "report_attempts": len(methods)}

    def _report_attempt(
        self, messages: list[AnyMessage], method: str
    ) -> tuple[AnalysisResult | None, str, str]:
        """One report request. Returns (result, error, retry_prompt); result is None on failure."""
        try:
            structured = self.model.with_structured_output(AnalysisResult, method=method, include_raw=True)
            out = structured.invoke(messages)
        except Exception as exc:  # provider/model errors (e.g. a truncated tool call) count as a failed attempt
            error = f"{type(exc).__name__}: {exc}"[:1000]
            return None, error, RETRY_PROMPT.format(error=error)
        if isinstance(out.get("parsed"), AnalysisResult):
            return out["parsed"], "", ""
        if out.get("parsing_error") is not None:
            error = str(out["parsing_error"])[:1000]
            return None, error, RETRY_PROMPT.format(error=error)
        raw = out.get("raw")
        rescued = _report_from_text(raw.content if isinstance(raw, AIMessage) else "")
        if rescued is not None:
            return rescued, "", ""
        return None, "the model replied in plain text without calling the AnalysisResult tool", NO_TOOL_CALL_PROMPT

    def verify(self, state: AgentState) -> dict[str, Any]:
        drafted = AnalysisResult.model_validate({k: state[k] for k in AnalysisResult.model_fields if k in state})
        request_text = "\n".join([state.get("question", ""), state.get("context") or ""])
        result = verify_result(drafted, state.get("tool_outputs", []), request_text)
        removed = len(drafted.findings) - len(result.findings)
        workspace = self._workspace(state)
        extra = [state["stop_reason"]] if state.get("stop_reason") else []
        result = result.model_copy(
            update={"charts": workspace.chart_paths(), "limitations": [*result.limitations, *extra]}
        )
        return {**result.model_dump(mode="json"), "removed_findings": removed}

    def _workspace(self, state: AgentState) -> Workspace:
        return Workspace.open(Path(state["workspace_root"]), timeout_s=self.settings.tool_timeout_s)


def _report_from_text(content: Any) -> AnalysisResult | None:
    """Accept a report the model wrote as JSON text instead of a tool call, if it validates."""
    text = content if isinstance(content, str) else ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return AnalysisResult.model_validate_json(text[start : end + 1])
    except ValidationError:
        return None


def _call_tool(available: dict[str, Any], name: str, args: dict[str, Any]) -> str:
    tool = available.get(name)
    if tool is None:
        return f"Error: unknown tool {name!r}. Available tools: {', '.join(available)}"
    try:
        return str(tool(**args))
    except TypeError as exc:
        return f"Error: invalid arguments for {name}: {exc}"
    except Exception as exc:  # report to the model instead of crashing the run
        log.exception("tool %s failed", name)
        return f"Error: {name} failed: {type(exc).__name__}: {exc}"
