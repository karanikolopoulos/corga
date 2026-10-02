import logging

from langgraph.runtime import Runtime

from corga.agent.state import Update, TurnState, recent
from corga.services.results import AnswerResult
from corga.agent.runtime.deps import Deps

log = logging.getLogger(__name__)


def ask(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Run the agent over the recent messages.

    Its state holds the stored window, plan and preferences. Its tool messages stay in its own run.

    Args:
        state: The turn's state.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The agent's last message as the result.
    """
    deps = runtime.context
    turn = {"window": state.window, "itinerary": state.itinerary, "preferences": state.preferences}

    out = deps.agent.invoke({"messages": recent(state.messages), **turn}, context=deps)
    answer = out["messages"][-1].text

    log.debug("ask, the answer: %s", answer)

    return {"result": AnswerResult(answer=answer)}
