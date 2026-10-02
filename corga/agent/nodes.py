"""Graph nodes. Each takes the state plus the runtime (whose context is `Deps`) and returns a partial state.

`result` is the turn's structured outcome: the handlers fill it, `respond` frames it (LLM) and renders it (template).
Every node starts with `turn = Turn(state, runtime.context)`, which names the state and holds the shared helpers.
"""

import json

from typing import Any, Literal

from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue

from corga.agent.deps import Deps
from corga.agent.turn import (
    Turn,
    dump,
    unique,
    refusal,
    need_info,
    resolve_window,
    merge_preferences,
    unsupported_topics,
)
from corga.agent.state import Intent, Result, AgentState, ParsedRequest
from corga.data.models import POI
from corga.data.timeutil import Interval, now
from corga.weather.models import HourlyWeather, WeatherUnavailable
from corga.weather.replan import wet_periods
from corga.agent.templates import render_result
from corga.planning.models import Itinerary, Preferences
from corga.planning.builder import (
    Replan,
    FeasibilityReport,
)
from corga.planning.candidates import matched_interests

Handler = Literal["answer", "recommend", "plan", "edit_plan", "weather", "refuse"]
ROUTES: dict[Intent, Handler] = {
    "question": "answer",
    "recommend": "recommend",
    "plan": "plan",
    "edit_plan": "edit_plan",
    "weather": "weather",
    "out_of_scope": "refuse",
}

RECENT_MESSAGES = 5  # the two previous turns plus this message, starting on a user message


def route(state: AgentState) -> Handler:
    """The handler for this turn; a request to invent or ignore the dataset is refused whatever its intent."""
    request: ParsedRequest = state["request"]
    return "refuse" if request.wants_fabrication else ROUTES[request.intent]


# --- what the result says about a place ---


def _fact(poi: POI, score: float | None) -> dict[str, Any]:
    """What the answer may say about a place; null opening hours stay null (unknown, not open).

    `score` is the search similarity, or None for a place the user named that the search did not return.
    """
    return {
        "id": poi.id,
        "score": round(score, 3) if score is not None else None,
        "name": poi.name,
        "category": poi.category,
        "setting": poi.setting,
        "description": poi.description,
        "opening_hours": poi.opening_hours.root if poi.opening_hours else None,
        "source_urls": poi.source_urls,
    }


def _item(poi: POI, prefs: Preferences) -> dict[str, Any]:
    """A recommended place and which of the interests it matches."""
    return {
        "id": poi.id,
        "name": poi.name,
        "category": poi.category,
        "setting": poi.setting,
        "matched": matched_interests(poi, prefs.interests),
        "family_suitable": poi.family_suitable,
    }


def _hour(h: HourlyWeather) -> dict[str, Any]:
    return {
        "time": f"{h.time:%a %H:%M}",
        "precipitation_mm": h.precipitation_mm,
        "temperature_c": h.temperature_c,
        "description": h.description,
    }


# --- nodes ---


def parse(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """Interpret the latest message (LLM), then fold it into the sticky window and preferences (code)."""
    turn = Turn(state, runtime.context)
    inputs = {"today": now().date().isoformat(), "context": turn.summary, "messages": turn.messages[-RECENT_MESSAGES:]}
    prompt: PromptValue = turn.deps.parse_prompt.invoke(inputs)
    request: ParsedRequest = turn.deps.structured(ParsedRequest).invoke(prompt)
    return {
        "request": request,
        "preferences": merge_preferences(turn.prefs, request),
        "window": resolve_window(turn.window, request),
    }


def answer(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """Facts about the places the user named plus the closest matches to the question, all from the dataset."""
    turn = Turn(state, runtime.context)
    question = turn.question
    named = turn.lookup(turn.request.requested_places)
    # TODO(calibrate): drop hits below a minimum similarity so off-topic questions cite nothing, instead of the
    # 4 nearest POIs; pick the threshold from MiniLM scores on on-topic vs off-topic questions (e.g. 'aquarium').
    hits = turn.search(question)
    scores = {hit.poi.id: hit.score for hit in hits}
    pois: list[POI] = unique([*named.places, *(hit.poi for hit in hits)])  # named first
    facts = [_fact(p, scores.get(p.id)) for p in pois]
    result: Result = {
        "kind": "answer",
        "question": question,
        "facts": facts,
        "citations": [f["id"] for f in facts],
        "unsupported_topics": unsupported_topics(question),
        "unknown_places": named.unknown,
    }
    return {"result": result}


def recommend(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """The top candidates for the preferences, open on the planned day when there is one."""
    turn = Turn(state, runtime.context)
    steps = turn.deps.steps
    pois: list[POI] = steps.places(steps.recommend.invoke({"preferences": turn.prefs, "day": turn.day}))
    items = [_item(p, turn.prefs) for p in pois]
    result: Result = {
        "kind": "recommendation",
        "preferences": dump(turn.prefs),
        "items": items,
        "citations": [i["id"] for i in items],
    }
    return {"result": result}


def plan(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """A new itinerary from the preferences, or a feasibility check of the places the user named."""
    turn = Turn(state, runtime.context)
    window, steps = turn.window, turn.deps.steps
    if window is None:
        return need_info("a date plus start and end time")
    if not turn.request.requested_places:
        candidates = [poi.id for poi in turn.candidates()]
        built: Itinerary = steps.build_itinerary.invoke({"window": window, "candidates": candidates})
        return turn.planned(built, {"kind": "plan"})

    named = turn.lookup(turn.request.requested_places)
    if named.unknown:
        return refusal("not_in_dataset", names=named.unknown)
    places = [poi.id for poi in named.places]
    report: FeasibilityReport = steps.check_requested.invoke({"places": places, "window": window})
    result: Result = {"kind": "plan", "feasibility": dump(report)}
    itinerary: Itinerary | None = report.itinerary or report.alternative
    return turn.planned(itinerary, result) if itinerary else {"result": result}


def edit_plan(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """Replace one stop, or, without a stop number, apply newly stated preferences to the whole plan.

    Unaffected stops and the window are kept either way.
    """
    turn = Turn(state, runtime.context)
    itinerary, index = turn.itinerary, turn.request.stop_index  # index is 1-based
    if itinerary is None:
        return need_info("an existing itinerary to edit")
    if index is None and turn.request.preference_updates:
        return _revise_for_preferences(turn, itinerary)
    if index is None or not 1 <= index <= len(itinerary.stops):
        return need_info(f"which stop to change (1-{len(itinerary.stops)})")

    args = {"itinerary": itinerary, "index": index, "preferences": turn.prefs}
    new: Itinerary | None = turn.deps.steps.replace_stop.invoke(args)
    if new is None:
        reason = "no alternative fits the time window and constraints"
        return {"result": {"kind": "edit_failed", "stop_index": index, "reason": reason}}
    result: Result = {"kind": "plan", "edited_stop": index, "replaced": itinerary.stops[index - 1].poi_id}
    return turn.planned(new, result)


def _revise_for_preferences(turn: Turn, itinerary: Itinerary) -> AgentState:
    """The plan rebuilt from its first stop that the updated preferences no longer select."""
    args = {"itinerary": itinerary, "preferences": turn.prefs}  # the preferences are already updated by `parse`
    replan: Replan = turn.deps.steps.replan_for_preferences.invoke(args)
    result: Result = {"kind": "plan", "adaptation": dump(replan, exclude={"itinerary"})}
    return turn.planned(replan.itinerary, result)


def weather(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """The forecast for the relevant hours and, when there is a plan, the plan adapted to rain."""
    turn = Turn(state, runtime.context)
    steps = turn.deps.steps
    start, end = turn.weather_range()
    try:
        forecast: dict[str, Any] = steps.weather_forecast.invoke({"start": start, "end": end})
    except WeatherUnavailable as exc:
        return {"result": {"kind": "weather_unavailable", "source": turn.deps.weather.source, "reason": str(exc)}}

    hours: list[HourlyWeather] = forecast["hours"]
    periods: list[Interval] = wet_periods(hours)
    result: Result = {
        "kind": "weather",
        "source": forecast["source"],
        "range": [start.isoformat(), end.isoformat()],
        "rain_expected": bool(periods),
        "wet_periods": [{"start": p.start.isoformat(), "end": p.end.isoformat()} for p in periods],
        "hours": [_hour(h) for h in hours],
    }
    if turn.itinerary is None:
        return {"result": result}
    args = {"itinerary": turn.itinerary, "wet_periods": periods, "preferences": turn.prefs}
    adaptation: Replan = steps.replan_for_rain.invoke(args)
    result["adaptation"] = dump(adaptation, exclude={"itinerary"})
    return turn.planned(adaptation.itinerary, result)


def refuse(state: AgentState) -> AgentState:
    return refusal("fabrication_or_ignore_dataset" if state["request"].wants_fabrication else "out_of_scope")


def respond(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """A few framing sentences from the LLM, followed by the deterministic rendering of the result."""
    turn = Turn(state, runtime.context)
    payload = json.dumps(turn.result, ensure_ascii=False, default=str)
    framing: BaseMessage = turn.deps.llm.invoke(turn.deps.respond_prompt.invoke({"result": payload}))
    body = render_result(turn.result, turn.deps.store)
    return {"messages": [AIMessage(content=f"{framing.text}\n\n{body}".strip())]}
