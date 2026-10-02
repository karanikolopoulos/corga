"""The first node of every turn: the LLM reads the message, the code folds it into the stored state."""

from langgraph.runtime import Runtime
from langchain_core.prompt_values import PromptValue

from corga.agent.state import Update, AgentState, ParsedRequest
from corga.data.timeutil import now
from corga.agent.templates import state_summary
from corga.agent.runtime.deps import Deps
from corga.agent.nodes.request import resolve_window, merge_preferences

RECENT_MESSAGES = 5  # the two previous turns plus this message, starting on a user message


def parse(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """Interpret the latest message (LLM), then fold it into the sticky window and preferences (code)."""
    deps = runtime.context
    inputs = {
        "today": now().date().isoformat(),
        "context": state_summary(state),
        "messages": state.messages[-RECENT_MESSAGES:],
    }
    prompt: PromptValue = deps.parse_prompt.invoke(inputs)
    request: ParsedRequest = deps.parser.invoke(prompt)
    return {
        "request": request,
        "preferences": merge_preferences(state.preferences, request),
        "window": resolve_window(state.window, request),
    }
