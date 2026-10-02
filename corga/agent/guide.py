"""The guide the nodes call: planning steps, dataset lookups and the weather tool, over one store, rules, retriever and
weather provider.

Each planning step is a traced method: the session trace records every call with its arguments and result. The
forecast goes through the weather tool and the search through the retriever, so both are recorded as well.
"""

from typing import Annotated, NamedTuple
from datetime import date, datetime
from collections.abc import Iterable

from langchain_core.tools import InjectedToolArg, tool

from corga.planning import checks, builder, candidates
from corga.data.store import POIStore
from corga.data.models import POI
from corga.agent.tracing import traced
from corga.data.timeutil import Interval
from corga.planning.rules import PlanningRules
from corga.weather.models import HourlyWeather, WeatherProvider
from corga.planning.models import Itinerary, TimeWindow, Preferences, ValidationResult
from corga.planning.replan import Replan, adapt_for_weather, adapt_for_preferences, replace_stop_for_preferences
from corga.planning.builder import FeasibilityReport
from corga.retrieval.search import Retriever


class Hit(NamedTuple):
    """A place the search returned, with its cosine similarity to the query (higher is closer)."""

    poi: POI
    score: float


class Lookup(NamedTuple):
    """Place names the user gave, split into dataset places and names the dataset does not have."""

    places: list[POI]  # in the order named, each once
    unknown: list[str]


def unique(pois: Iterable[POI]) -> list[POI]:
    """The places in order, each once."""
    return list({poi.id: poi for poi in pois}.values())


@tool
def weather_forecast(
    start: datetime, end: datetime, provider: Annotated[WeatherProvider, InjectedToolArg]
) -> list[HourlyWeather]:
    """Hourly forecast for Corfu from `start` to `end` (Europe/Athens)."""
    return provider.hourly(start, end)


class Guide:
    """Planning steps, dataset lookups and the forecast, built once per application and held by `Deps`."""

    def __init__(self, store: POIStore, rules: PlanningRules, retriever: Retriever, weather: WeatherProvider):
        self.store = store
        self.rules = rules
        self.retriever = retriever
        self.weather = weather

    # --- planning steps: each call is recorded in the trace with its arguments and result ---

    @traced
    def select_candidates(self, prefs: Preferences) -> list[POI]:
        """preferences → the places passing them, ranked by the interests."""
        return candidates.select_candidates(self.store.all(), prefs, self.retriever.rank)

    @traced
    def recommendations(self, prefs: Preferences, day: date | None) -> list[POI]:
        """preferences, day → the best places open that day."""
        return candidates.recommendations(self.store.all(), prefs, day, self.retriever.rank)

    @traced
    def build_itinerary(self, window: TimeWindow, pois: list[POI]) -> Itinerary:
        """window, candidates → a greedy itinerary."""
        return builder.build_itinerary(window, pois, self.rules)

    @traced
    def replace_stop(self, itinerary: Itinerary, index: int, prefs: Preferences) -> Replan:
        """itinerary, stop number, preferences → that stop replaced, or the itinerary unchanged if nothing fits."""
        return replace_stop_for_preferences(
            itinerary,
            index,
            self.store.all(),
            prefs,
            rules=self.rules,
            ranker=self.retriever.rank,
        )

    @traced
    def replan_for_rain(self, itinerary: Itinerary, wet_periods: list[Interval], prefs: Preferences) -> Replan:
        """itinerary, wet periods, preferences → rebuilt from indoor places from the first exposed stop."""
        return adapt_for_weather(
            itinerary,
            wet_periods,
            indoor_candidates=lambda: self.select_candidates(prefs.indoor()),
            rules=self.rules,
        )

    @traced
    def replan_for_preferences(self, itinerary: Itinerary, prefs: Preferences) -> Replan:
        """itinerary, preferences → rebuilt from the first stop the preferences no longer select."""
        return adapt_for_preferences(itinerary, self.select_candidates(prefs), self.rules)

    @traced
    def check_requested(self, places: list[POI], window: TimeWindow) -> FeasibilityReport:
        """named places, window → whether any visiting order fits, why not, and an alternative."""
        return builder.check_requested(places, window, self.rules)

    @traced
    def validate(self, itinerary: Itinerary) -> ValidationResult:
        """itinerary → the hard-constraint checks."""
        return checks.validate(itinerary, self.rules)

    # --- dataset lookups and the weather tool ---

    def search(self, query: str) -> list[Hit]:
        """The dataset places closest to the query, best first, with their similarity scores."""
        documents = self.retriever.invoke(query)

        return [Hit(self.store.get(doc.metadata["id"]), doc.metadata["score"]) for doc in documents]

    def lookup(self, names: list[str]) -> Lookup:
        """Match place names against the dataset."""
        found = {name: self.store.find(name) for name in names}
        places = unique(poi for poi in found.values() if poi is not None)

        return Lookup(places, [name for name, poi in found.items() if poi is None])

    def forecast(self, start: datetime, end: datetime) -> list[HourlyWeather]:
        """The weather tool's hourly forecast for the window."""
        return weather_forecast.invoke({"start": start, "end": end, "provider": self.weather})
