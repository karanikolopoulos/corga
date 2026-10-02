"""The visitor's itinerary: a new plan, a check of named places, edits, and the plan adapted to the weather.

Every itinerary a use case returns carries its validation.
"""

from corga.domain.weather import HourlyWeather, wet_periods
from corga.services.steps import PlanningSteps
from corga.domain.timeutil import Interval
from corga.services.places import PlaceFinder
from corga.services.results import (
    Adaptation,
    PlanResult,
    PlannedResult,
    RefusalResult,
    WeatherResult,
    NeedInfoResult,
)
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences
from corga.domain.planning.replan import Replan

WINDOW_MISSING = "a date plus start and end time"
ITINERARY_MISSING = "an existing itinerary to edit"


class Itineraries:
    """The itinerary use cases, over the planning steps and the place finder; held by `Deps`."""

    def __init__(self, steps: PlanningSteps, places: PlaceFinder):
        self.steps = steps
        self.places = places

    def plan_itinerary(self, window: TimeWindow | None, prefs: Preferences) -> PlanResult | NeedInfoResult:
        """A new itinerary for the window from the places the preferences select."""
        if window is None:
            return NeedInfoResult(missing=WINDOW_MISSING)
        built = self.steps.build_itinerary(window, self.steps.select_candidates(prefs))
        return self._validated(PlanResult(itinerary=built))

    def check_places(self, window: TimeWindow | None, names: list[str]) -> PlanResult | NeedInfoResult | RefusalResult:
        """Whether the named places fit the window in some order, why not, and the closest feasible alternative."""
        if window is None:
            return NeedInfoResult(missing=WINDOW_MISSING)
        lookup = self.places.lookup(names)
        if lookup.unknown:
            return RefusalResult(reason="not_in_dataset", names=lookup.unknown)
        report = self.steps.check_requested(lookup.places, window)
        return self._validated(PlanResult(feasibility=report, itinerary=report.itinerary or report.alternative))

    def revise_itinerary(self, itinerary: Itinerary | None, prefs: Preferences) -> PlanResult | NeedInfoResult:
        """The plan rebuilt from its first stop that the preferences no longer select; the window is kept."""
        if itinerary is None:
            return NeedInfoResult(missing=ITINERARY_MISSING)
        return self._replanned(self.steps.replan_for_preferences(itinerary, prefs))

    def change_stop(
        self, itinerary: Itinerary | None, index: int | None, prefs: Preferences
    ) -> PlanResult | NeedInfoResult:
        """Stop `index` (1-based) replaced by a place matching the preferences; the other stops and the window kept."""
        if itinerary is None:
            return NeedInfoResult(missing=ITINERARY_MISSING)
        if index is None or not 1 <= index <= len(itinerary.stops):
            return NeedInfoResult(missing=f"which stop to change (1-{len(itinerary.stops)})")
        return self._replanned(self.steps.replace_stop(itinerary, index, prefs))

    def report_weather(
        self, hours: list[HourlyWeather], span: Interval, source: str, itinerary: Itinerary | None, prefs: Preferences
    ) -> WeatherResult:
        """The forecast for `span` and, when there is a plan, the plan adapted to its rain."""
        periods = wet_periods(hours)
        result = WeatherResult(source=source, range=span, wet_periods=periods, hours=hours)
        if itinerary is None:
            return result
        replan = self.steps.replan_for_rain(itinerary, periods, prefs)
        result.itinerary = replan.itinerary
        result.adaptation = Adaptation.model_validate(replan)
        return self._validated(result)

    def _replanned(self, replan: Replan) -> PlanResult:
        """An edited plan (or the unchanged one, with the note saying why) as a validated result."""
        return self._validated(PlanResult(itinerary=replan.itinerary, adaptation=Adaptation.model_validate(replan)))

    def _validated[R: PlannedResult](self, result: R) -> R:
        """The result with its itinerary's validation, when it carries an itinerary."""
        if result.itinerary is not None:
            result.validation = self.steps.validate(result.itinerary)
        return result
