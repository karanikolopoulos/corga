import logging
import datetime as dt

from datetime import date, datetime, timedelta

from langchain.tools import ToolRuntime, tool
from langchain.agents import AgentState

from corga.agent.request import Stated, Interest, SettingChoice, preference_updates
from corga.domain.timeutil import Interval, at, duration_text
from corga.agent.runtime.deps import Deps
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences

log = logging.getLogger(__name__)


class ForecastRangeError(ValueError):
    """The hours to forecast cannot be read from the request.

    No stored window or plan, a stop the plan does not have, or no range given.
    """


def forecast_span(
    window: TimeWindow | None,
    itinerary: Itinerary | None,
    now: datetime,
    *,
    stop: int | None = None,
    visit_window: bool = False,
    day: date | None = None,
    next_hours: int | None = None,
) -> Interval:
    """Find the hours a forecast covers.

    Give one of `stop`, `visit_window`, `day` or `next_hours`.

    Args:
        window: The stored window.
        itinerary: The stored plan.
        now: The current time.
        stop: A stop of the stored plan (1 is the first).
        visit_window: True for the stored window.
        day: A whole day.
        next_hours: The hours from now.

    Returns:
        The hours to forecast.

    Raises:
        ForecastRangeError: None is given, or the stored state lacks what it needs.
    """
    if visit_window:
        if window is None:
            raise ForecastRangeError("no time window is stored")

        return window.span

    if stop is not None:
        if itinerary is None:
            raise ForecastRangeError("no plan is stored")
        if not itinerary.has_stop(stop):
            raise ForecastRangeError(f"the plan has stops 1-{len(itinerary.stops)}")

        return itinerary.stops[stop - 1].span

    if day is not None:
        return Interval(at(day, "00:00"), at(day, "00:00") + timedelta(days=1))

    if next_hours is None:
        raise ForecastRangeError("give one of stop, visit_window, date or next_hours")

    this_hour = now.replace(minute=0, second=0, microsecond=0)

    return Interval(this_hour, this_hour + timedelta(hours=next_hours))


class ToolState(AgentState):
    """The agent's state as the tools read it.

    The loop's messages plus the turn's stored window, plan and preferences.
    """

    window: TimeWindow | None
    itinerary: Itinerary | None
    preferences: Preferences


@tool(parse_docstring=True)
def search(query: str, runtime: ToolRuntime[Deps, ToolState]) -> str:
    """Search the descriptions of the 50 places in the Corfu dataset.

    Return the closest ones, best first. Use it for a question about what a place is, what it has or what it is known
    for, and to look up a place the user names, including its opening hours. Each result gives the place's id in square
    brackets, its category, setting, suitability for children, suggested visit time, fixture opening hours for every
    weekday and description. The dataset holds no tickets, prices, bookings, accessibility details or live opening
    status, so another search will not find them. Not for recommendations by interest (use find_places) or for the
    weather (use forecast).

    Args:
        query: What to look for, in words: a place's name, or what the user asks about.
        runtime: The dependencies and the turn's state, given by LangChain.

    Returns:
        The places, one line each.
    """
    hits = runtime.context.places.search(query)

    log.info("agent tool: search(%r) → %s", query, [hit.poi.id for hit in hits])

    return runtime.context.prompts["places"].render(places=[hit.poi for hit in hits]).strip()


@tool(parse_docstring=True)
def find_places(
    runtime: ToolRuntime[Deps, ToolState],
    *,
    interests: list[Interest] | None = None,
    child: Stated = "not_stated",
    avoid_museums: Stated = "not_stated",
    setting: SettingChoice = "not_stated",
    date: dt.date | None = None,
) -> str:
    """Recommend places from the dataset by interest and constraint.

    The best matches come first. Places are filtered on their metadata, not searched by text, and ranked for the
    interests. The result holds only the best few matches (at most five), not every place that fits: a place missing
    from it may still match or be open. Use it for "what should I see" questions. To check one place, including its
    opening hours, use search. Results read like search results.

    Args:
        runtime: The dependencies and the turn's state, given by LangChain.
        interests: Topics from history, venetian, museums, art, nature, food, architecture. Empty uses the stored ones.
        child: "yes" if a child comes along, "no" if not. "not_stated" uses the stored preference.
        avoid_museums: "yes" to leave museums out, "no" to allow them. "not_stated" uses the stored preference.
        setting: "indoor" or "outdoor" places, "any" for either. "not_stated" uses the stored preference.
        date: A day (YYYY-MM-DD). Only places open that day in the fixture hours are recommended.

    Returns:
        The places, one line each, or "no place matches".
    """
    stated = preference_updates(interests or [], child, avoid_museums, setting)
    prefs = runtime.state["preferences"].merged(stated)
    places = runtime.context.places.recommend(prefs, date)

    log.info(
        "agent tool: find_places(%s, %s) → %s", prefs.model_dump(exclude_defaults=True), date, [p.id for p in places]
    )

    return runtime.context.prompts["places"].render(places=places).strip()


@tool(parse_docstring=True)
def forecast(
    runtime: ToolRuntime[Deps, ToolState],
    stop: int | None = None,
    visit_window: bool = False,
    date: dt.date | None = None,
    next_hours: int | None = None,
) -> str:
    """Get the hourly weather forecast for Corfu.

    For a stop of the current plan, the visit's time window, a day or the next hours. Give exactly one of the arguments.
    For a question with no time ("what is the weather like?"), give next_hours=24. Code reads the times from the stored
    plan or the clock, so do not compute them. The result gives the forecast's id in square brackets and each hour's
    rain in mm and conditions. With a stop, also the stop's id, name and times. A day the provider does not forecast
    returns an error saying so. Use it only when the question is about the weather or rain. It says nothing about places
    or opening hours.

    Args:
        runtime: The dependencies and the turn's state, given by LangChain.
        stop: The number of a stop in the current plan (1 is the first). The forecast covers that visit.
        visit_window: True for the visit's whole time window (the current plan's window).
        date: A day (YYYY-MM-DD). The forecast covers the whole day.
        next_hours: The next N hours from now.

    Returns:
        The forecast, one line per hour.
    """
    context, state = runtime.context, runtime.state
    span = forecast_span(
        state["window"],
        state["itinerary"],
        context.now(),
        stop=stop,
        visit_window=visit_window,
        day=date,
        next_hours=next_hours,
    )

    report = context.weather.report(span)

    visit = state["itinerary"].stops[stop - 1] if stop is not None else None
    text = context.prompts["forecast"].render(report=report, stop=stop, visit=visit).strip()

    log.info(
        "tool: forecast %s %s for %s → rain %s",
        context.weather.source,
        f"{span.start:%Y-%m-%d %H:%M}",
        duration_text(span.end - span.start),
        report.rain_expected,
    )

    return text
