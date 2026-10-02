import logging

from typing import Literal

from pydantic import Field, BaseModel
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage

from corga.agent.state import Path, Update, TurnState, recent
from corga.agent.runtime.deps import Deps

log = logging.getLogger(__name__)


class Route(BaseModel):
    """The path a message takes: plan, ask, attack or out of scope."""

    path: Path = Field(description="The path the latest message takes.")


NEXT = {"plan": "extract", "ask": "ask"}  # attack and out_of_scope go to refuse


def route(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Pick the latest message's path with the LLM.

    Args:
        state: The turn's state.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The path: plan, ask, attack or out of scope.
    """
    deps = runtime.context
    path = deps.structured(Route).invoke([SystemMessage(deps.prompts["route"].render()), *recent(state.messages)]).path

    log.info("route: %s", path)
    log.debug("route, the message: %s", state.question)

    return {"path": path}


def choose(state: TurnState) -> Literal["extract", "ask", "refuse"]:
    return NEXT.get(state.path, "refuse")
