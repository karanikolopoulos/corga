import logging

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import (
    ModelRequest,
    ToolErrorMiddleware,
    ModelCallLimitMiddleware,
    dynamic_prompt,
)
from langchain_core.language_models import BaseChatModel

from corga.agent.tools import ToolState, ForecastRangeError, search, forecast, find_places
from corga.domain.weather import WeatherUnavailable
from corga.agent.runtime.deps import Deps

log = logging.getLogger(__name__)

MODEL_CALLS = 5  # the rounds of tool calls and the answer, after which the run raises


@dynamic_prompt
def plan_prompt(request: ModelRequest) -> str:
    deps: Deps = request.runtime.context

    return deps.prompts["ask"].render(today=deps.now().date(), itinerary=request.state["itinerary"])


def on_tool_error(error: Exception, request: ToolCallRequest) -> str | None:
    """Turn a tool's failure into the message the model reads.

    Args:
        error: What the tool raised.
        request: The tool call that failed.

    Returns:
        The message for a forecast range that cannot be read or an unavailable forecast. None for any other error, which
        then propagates.
    """
    if not isinstance(error, (ForecastRangeError, WeatherUnavailable)):
        return None

    log.warning("agent tool %s failed: %s", request.tool_call["name"], error)

    return f"{request.tool_call['name']} failed: {error}"


def build_agent(llm: BaseChatModel) -> CompiledStateGraph:
    return create_agent(
        llm,
        tools=[search, find_places, forecast],
        context_schema=Deps,
        state_schema=ToolState,
        middleware=[
            plan_prompt,
            ToolErrorMiddleware(on_tool_error),
            ModelCallLimitMiddleware(run_limit=MODEL_CALLS, exit_behavior="error"),
        ],
    )
