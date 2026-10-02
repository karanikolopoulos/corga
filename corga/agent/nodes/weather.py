"""The weather node: the forecast from the weather tool and, when there is a plan, the plan adapted to rain."""

from langgraph.runtime import Runtime

from corga.agent.state import Update, AgentState
from corga.agent.results import Adaptation, WeatherResult, WeatherUnavailableResult
from corga.data.timeutil import Interval, now
from corga.weather.models import WeatherUnavailable, wet_periods
from corga.agent.runtime.deps import Deps
from corga.agent.nodes.request import weather_range
from corga.agent.nodes.results import planned


def weather(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """The forecast for the relevant hours and, when there is a plan, the plan adapted to rain."""
    guide = runtime.context.guide
    start, end = weather_range(state, now())
    try:
        forecast = guide.forecast(start, end)
    except WeatherUnavailable as exc:
        return {"result": WeatherUnavailableResult(source=guide.weather.source, reason=str(exc))}

    periods = wet_periods(forecast)
    result = WeatherResult(source=guide.weather.source, range=Interval(start, end), wet_periods=periods, hours=forecast)
    if state.itinerary is None:
        return {"result": result}
    replan = guide.replan_for_rain(state.itinerary, periods, state.preferences)
    result.itinerary = replan.itinerary
    result.adaptation = Adaptation.model_validate(replan)
    return planned(guide, result, previous=state.itinerary)
