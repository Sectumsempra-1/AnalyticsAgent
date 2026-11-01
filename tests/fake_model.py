"""A scripted chat model for graph tests: no network, deterministic."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from analytics_agent.schemas import AnalysisResult


def tool_call(name: str, call_id: str = "c1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedModel(BaseChatModel):
    """Returns `turns` in order for agent calls (then a plain 'done'), and `reports` in
    order for structured-output calls. A report may be an Exception to simulate a bad output."""

    turns: list[AIMessage] = Field(default_factory=list)
    reports: list[AnalysisResult | Exception] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.turns.pop(0) if self.turns else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":  # type: ignore[override]
        self.bound_tools = [t.__name__ for t in tools]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        def respond(_: Any) -> AnalysisResult:
            item = self.reports.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(respond)
