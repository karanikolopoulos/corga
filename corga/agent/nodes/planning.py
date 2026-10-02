import logging

from typing import Literal

from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage

from corga.agent.state import Update, TurnState, recent
from corga.agent.request import PlanRequest, change_of, resolve_window, merge_preferences
from corga.services.results import Result, PlanResult, NeedInfoResult, PreferencesResult
from corga.agent.runtime.deps import Deps
from corga.services.itineraries import WINDOW_MISSING
from corga.domain.planning.models import Itinerary, TimeWindow

log = logging.getLogger(__name__)


def span_text(window: TimeWindow | None) -> str:
    return f"{window.start:%a %d %b %H:%M}-{window.end:%H:%M}" if window else "none"


def remember(result: Result, previous: Itinerary | None) -> Update:
    """Return the result and the plan the conversation keeps.

    Args:
        result: The turn's result.
        previous: The stored plan.

    Returns:
        The result, with the result's itinerary when it passed validation, else `previous`.
    """
    carries_a_plan = isinstance(result, PlanResult) and result.itinerary is not None
    if not carries_a_plan:
        return {"result": result}

    kept = result.itinerary if result.validation.ok else previous

    return {"itinerary": kept, "result": result}


def extract(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Read the latest message into a `PlanRequest` with the LLM.

    The stored plan is in its prompt, so the message can refer to its stops.

    Args:
        state: The turn's state.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The plan request.
    """
    deps = runtime.context
    system = deps.prompts["extract"].render(today=deps.now().date(), itinerary=state.itinerary)

    request = deps.structured(PlanRequest).invoke([SystemMessage(system), *recent(state.messages)])

    log.info("plan request: %s", request.model_dump(mode="json", exclude_defaults=True))

    return {"plan_request": request}


def resolve(state: TurnState) -> Update:
    """Fold the request into the stored window and preferences.

    Args:
        state: The turn's state, with the plan request.

    Returns:
        The window and preferences, and a result when no plan can be built.
    """
    request = state.plan_request
    window = resolve_window(state.window, request)
    prefs = merge_preferences(state.preferences, request)

    if window != state.window:
        log.info("window: %s → %s", span_text(state.window), span_text(window))

    if prefs != state.preferences:
        log.info("preferences: %s → %s", state.preferences.model_dump(), prefs.model_dump())

    update: Update = {"window": window, "preferences": prefs}

    if request.only_preferences and state.itinerary is None:
        log.info("preferences stored; no plan to change")
        update["result"] = PreferencesResult(preferences=prefs)
    elif window is None:
        log.info("need info: %s", WINDOW_MISSING)
        update["result"] = NeedInfoResult(missing=WINDOW_MISSING)

    return update


def after_resolve(state: TurnState) -> Literal["respond", "forecast", "build"]:
    """Choose the next node after `resolve`.

    Args:
        state: The turn's state.

    Returns:
        `respond` when the result is set, `forecast` when the message asks for it, else `build`.
    """
    if state.result is not None:
        return "respond"
    if "forecast" in state.plan_request.sources:
        return "forecast"

    return "build"


def forecast(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Fetch the forecast over the visit's window. No LLM reads it.

    Args:
        state: The turn's state.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The forecast report.
    """
    report = runtime.context.weather.report(state.window.span)

    rain = [f"{period.start:%H:%M}-{period.end:%H:%M}" for period in report.wet_periods]
    hours = [(f"{h.time:%H:%M}", h.precipitation_mm, h.weather_code) for h in report.hours]
    log.info("forecast: %s %s, rain %s", report.source, span_text(state.window), rain or "none")
    log.debug("forecast hours: %s", hours)

    return {"forecast": report}


def build(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Check the named places, or change the stored plan as the request asks.

    Args:
        state: The turn's state.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The result and the plan the conversation keeps.
    """
    itineraries, request = runtime.context.itineraries, state.plan_request
    if request.places:
        result = itineraries.check_places(state.window, request.places)
    else:
        change = change_of(request)
        result = itineraries.plan(state.window, state.preferences, state.itinerary, change, state.forecast)

    return remember(result, previous=state.itinerary)
