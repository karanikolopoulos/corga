import logging

from typing import Literal

from langgraph.types import Command
from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

from corga.agent.state import TurnState
from corga.domain.weather import WeatherUnavailable
from corga.services.results import NeedInfoResult, WeatherUnavailableResult
from corga.agent.runtime.deps import Deps

log = logging.getLogger(__name__)

REPHRASE = "a message it can read: please rephrase it"


def rephrase(state: TurnState, error: NodeError) -> Command[Literal["respond"]]:
    log.warning("%s failed, asking to rephrase: %s", error.node, error.error)

    return Command(goto="respond", update={"result": NeedInfoResult(missing=REPHRASE)})


def no_answer(state: TurnState, error: NodeError) -> Command[Literal["respond"]]:
    """Ask to rephrase when the agent did not answer within its model calls.

    Args:
        state: The turn's state.
        error: The node's failure.

    Returns:
        A jump to `respond` with the rephrase request.

    Raises:
        Exception: Any other error, unchanged.
    """
    if not isinstance(error.error, ModelCallLimitExceededError):
        raise error.error

    log.warning("the agent did not answer: %s", error.error)

    return Command(goto="respond", update={"result": NeedInfoResult(missing=REPHRASE)})


def forecast_unavailable(state: TurnState, error: NodeError, runtime: Runtime[Deps]) -> Command[Literal["respond"]]:
    """Keep the plan and report that the forecast for its window failed.

    Args:
        state: The turn's state.
        error: The node's failure.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        A jump to `respond` with the failed forecast.

    Raises:
        Exception: Any error but `WeatherUnavailable`, unchanged.
    """
    if not isinstance(error.error, WeatherUnavailable):
        raise error.error

    log.warning("forecast unavailable, the plan is kept: %s", error.error)

    result = WeatherUnavailableResult(source=runtime.context.weather.source, reason=str(error.error))

    return Command(goto="respond", update={"result": result})
