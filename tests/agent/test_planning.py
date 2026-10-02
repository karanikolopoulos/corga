import pytest

from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langchain_core.exceptions import OutputParserException

from corga.agent.state import TurnState
from corga.agent.request import PlanRequest
from corga.domain.weather import WeatherUnavailable
from corga.services.results import NeedInfoResult, WeatherUnavailableResult
from corga.agent.nodes.errors import REPHRASE, rephrase, forecast_unavailable
from corga.agent.nodes.planning import resolve, after_resolve


def run(deps):
    return Runtime(context=deps)


def test_resolve_asks_for_a_window():
    update = resolve(TurnState(plan_request=PlanRequest(interests=["history"], replace_stops=[2])))

    assert update["result"] == NeedInfoResult(missing="a date plus start and end time")
    assert after_resolve(TurnState(result=update["result"])) == "respond"


def test_the_error_handlers_answer_in_respond(make_deps):
    unreadable = rephrase(TurnState(), NodeError(node="extract", error=OutputParserException("no PlanRequest")))

    assert (unreadable.goto, unreadable.update) == ("respond", {"result": NeedInfoResult(missing=REPHRASE)})

    error = NodeError(node="forecast", error=WeatherUnavailable("fixture has no data"))
    unavailable = forecast_unavailable(TurnState(), error, run(make_deps()))

    assert unavailable.update == {"result": WeatherUnavailableResult(source="fixture", reason="fixture has no data")}

    with pytest.raises(KeyError):
        forecast_unavailable(TurnState(), NodeError(node="forecast", error=KeyError("bug")), run(make_deps()))
