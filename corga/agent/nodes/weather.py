"""The weather node: the forecast from the weather tool and, when there is a plan, the plan adapted to rain."""

from langgraph.runtime import Runtime

from corga.agent.state import Update, AgentState
from corga.agent.tools import weather_forecast
from corga.agent.request import weather_range
from corga.domain.weather import WeatherUnavailable
from corga.domain.timeutil import Interval
from corga.services.results import WeatherUnavailableResult
from corga.agent.runtime.deps import Deps
from corga.agent.nodes.results import remember


def weather(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """The forecast for the relevant hours from the weather tool and, when there is a plan, the plan adapted to rain."""
    deps = runtime.context
    source = deps.weather.source
    start, end = weather_range(state, deps.now())
    try:
        hours = weather_forecast.invoke({"start": start, "end": end, "provider": deps.weather})
    except WeatherUnavailable as exc:
        return {"result": WeatherUnavailableResult(source=source, reason=str(exc))}
    result = deps.itineraries.report_weather(hours, Interval(start, end), source, state.itinerary, state.preferences)
    return remember(result, previous=state.itinerary)
