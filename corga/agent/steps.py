"""The operations the agent performs, each recorded in the trace: one tool (weather) and named planning steps.

A step is a named LangChain run: invoked inside a node, its input, output, timing and errors reach the session's
callbacks. Inputs and outputs are kept small (ids, codes, models of a few stops), never the whole dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from datetime import datetime
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import Runnable, RunnableLambda

from corga.data.models import POI
from corga.weather.models import WeatherProvider
from corga.weather.replan import adapt_for_weather
from corga.planning.checks import validate
from corga.planning.models import Itinerary, TimeWindow, Preferences, ValidationResult
from corga.planning.builder import (
    Replan,
    FeasibilityReport,
    build_itinerary,
    check_requested,
    replan_from_first_misfit,
    replace_stop_for_preferences,
)
from corga.planning.candidates import recommendations, select_candidates

if TYPE_CHECKING:
    from corga.agent.deps import Deps

STEP_TAG = "step"


def step(name: str, fn: Callable[[Any], Any]) -> Runnable:
    """`fn` as a named run in the trace."""
    return RunnableLambda(fn, name=name).with_config(tags=[STEP_TAG])


def weather_tool(provider: WeatherProvider) -> BaseTool:
    """The weather provider as a LangChain tool, so each call is recorded as a tool run."""

    @tool("weather_forecast")
    def weather_forecast(start: datetime, end: datetime) -> dict[str, Any]:
        """Hourly forecast for Corfu from `start` to `end` (Europe/Athens), and where it came from."""
        return {"source": provider.source, "hours": provider.hourly(start, end)}

    return weather_forecast


class Steps:
    """The traced operations, bound to one set of dependencies (read at call time, so they can be swapped)."""

    def __init__(self, deps: Deps):
        self._deps = deps
        self.weather_forecast = weather_tool(deps.weather)
        self.retrieve = step("retrieve", self._retrieve)
        self.select_candidates = step("select_candidates", self._select_candidates)
        self.recommend = step("recommend", self._recommend)
        self.build_itinerary = step("build_itinerary", self._build_itinerary)
        self.replace_stop = step("replace_stop", self._replace_stop)
        self.replan_for_rain = step("replan_for_rain", self._replan_for_rain)
        self.replan_for_preferences = step("replan_for_preferences", self._replan_for_preferences)
        self.check_requested = step("check_requested", self._check_requested)
        self.validate = step("validate", self._validate)

    def _retrieve(self, question: str) -> list[tuple[str, float]]:
        """question → (POI id, similarity) pairs, best first."""
        return [(doc.metadata["id"], doc.metadata["score"]) for doc in self._deps.retriever.search(question)]

    def _select_candidates(self, prefs: Preferences) -> list[str]:
        """preferences → ids of the places passing them, ranked by the interests."""
        deps = self._deps
        return [poi.id for poi in select_candidates(deps.store.all(), prefs, deps.retriever.rank)]

    def _recommend(self, args: dict[str, Any]) -> list[str]:
        """preferences, day → ids of the best places open that day."""
        deps, day = self._deps, args.get("day")
        prefs: Preferences = args["preferences"]
        return [poi.id for poi in recommendations(deps.store.all(), prefs, day, deps.retriever.rank)]

    def _build_itinerary(self, args: dict[str, Any]) -> Itinerary:
        """window, candidate ids → a greedy itinerary."""
        store = self._deps.store
        window: TimeWindow = args["window"]
        return build_itinerary(window, [store.get(pid) for pid in args["candidates"]], self._deps.rules)

    def _replace_stop(self, args: dict[str, Any]) -> Itinerary | None:
        """itinerary, stop number, preferences → the itinerary with that stop replaced, or None."""
        deps = self._deps
        return replace_stop_for_preferences(
            args["itinerary"],
            args["index"],
            deps.store.all(),
            args["preferences"],
            store=deps.store,
            rules=deps.rules,
            ranker=deps.retriever.rank,
        )

    def _replan_for_rain(self, args: dict[str, Any]) -> Replan:
        """itinerary, wet periods, preferences → rebuilt from indoor places from the first exposed stop."""
        deps, prefs = self._deps, args["preferences"]
        return adapt_for_weather(
            args["itinerary"],
            args["wet_periods"],
            indoor_candidates=lambda: self.places(self.select_candidates.invoke(prefs.indoor())),
            store=deps.store,
            rules=deps.rules,
        )

    def _replan_for_preferences(self, args: dict[str, Any]) -> Replan:
        """itinerary, preferences → rebuilt from the first stop the preferences no longer select."""
        deps = self._deps
        candidates = self.places(self.select_candidates.invoke(args["preferences"]))
        selected = {poi.id for poi in candidates}
        replan = replan_from_first_misfit(
            args["itinerary"], lambda stop: stop.poi_id in selected, lambda: candidates, deps.store, deps.rules
        )
        note = (
            f"Replaced {len(replan.replaced)} stop(s) that no longer match the preferences."
            if replan.changed
            else "Every stop still matches the preferences."
        )
        return replan.model_copy(update={"note": note})

    def _check_requested(self, args: dict[str, Any]) -> FeasibilityReport:
        """named place ids, window → whether any visiting order fits, why not, and an alternative."""
        deps = self._deps
        return check_requested(self.places(args["places"]), args["window"], deps.store, deps.rules)

    def _validate(self, itinerary: Itinerary) -> ValidationResult:
        """itinerary → the hard-constraint checks."""
        return validate(itinerary, self._deps.store, self._deps.rules)

    def places(self, ids: list[str]) -> list[POI]:
        """The dataset places for `ids`, in order."""
        return [self._deps.store.get(pid) for pid in ids]
