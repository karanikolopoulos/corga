"""The planning steps: the domain's planning functions over the dataset, the rules and the ranking of interests.

Each step is a traced method: the session trace records every call with its arguments and result.
"""

from datetime import date

from corga.tracing import traced
from corga.domain.places import POI
from corga.services.ports import PlaceCatalog
from corga.domain.planning import checks, builder, candidates
from corga.domain.timeutil import Interval
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences, ValidationResult
from corga.domain.planning.replan import Replan, adapt_for_weather, adapt_for_preferences, replace_stop_for_preferences
from corga.domain.planning.builder import FeasibilityReport
from corga.domain.planning.candidates import Ranker


class PlanningSteps:
    """The planning steps over one dataset, set of rules and ranking; built once per application."""

    def __init__(self, store: PlaceCatalog, rules: PlanningRules, ranker: Ranker):
        self.store = store
        self.rules = rules
        self.ranker = ranker

    @traced
    def select_candidates(self, prefs: Preferences) -> list[POI]:
        """preferences → the places passing them, ranked by the interests."""
        return candidates.select_candidates(self.store.all(), prefs, self.ranker)

    @traced
    def recommendations(self, prefs: Preferences, day: date | None) -> list[POI]:
        """preferences, day → the best places open that day."""
        return candidates.recommendations(self.store.all(), prefs, day, self.ranker)

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
            ranker=self.ranker,
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
