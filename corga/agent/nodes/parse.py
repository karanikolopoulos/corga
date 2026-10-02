"""The first node of every turn: the LLM reads the message, the code folds it into the stored state."""

from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage

from corga.agent.state import Update, AgentState, ParsedRequest
from corga.agent.request import resolve_window, merge_preferences
from corga.agent.runtime.deps import Deps

RECENT_MESSAGES = 5  # the two previous turns plus this message, starting on a user message


def parse(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """Interpret the latest message (LLM), then fold it into the sticky window and preferences (code)."""
    deps = runtime.context
    system = deps.prompts["parse"].render(today=deps.now().date().isoformat(), state=state)
    request: ParsedRequest = deps.parser.invoke([SystemMessage(system), *state.messages[-RECENT_MESSAGES:]])
    return {
        "request": request,
        "preferences": merge_preferences(state.preferences, request),
        "window": resolve_window(state.window, request),
    }
