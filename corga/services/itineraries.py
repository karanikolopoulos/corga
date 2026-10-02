import logging

from dataclasses import replace

from corga.domain.weather import WeatherReport
from corga.domain.planning import checks, refill, builder
from corga.services.places import PlaceFinder
from corga.services.results import (
    Adaptation,
    PlanResult,
    RefusalResult,
    NeedInfoResult,
)
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences
from corga.domain.planning.refill import NO_CHANGE, Change

log = logging.getLogger(__name__)

WINDOW_MISSING = "a date plus start and end time"
ITINERARY_MISSING = "an existing itinerary to edit"


def need_info(missing: str) -> NeedInfoResult:
    """Log and return a request for missing information.

    Args:
        missing: What is missing, in words.

    Returns:
        The result asking for it.
    """
    log.info("need info: %s", missing)

    return NeedInfoResult(missing=missing)


class Itineraries:
    def __init__(self, places: PlaceFinder, rules: PlanningRules):
        self.places = places
        self.rules = rules

    def plan(
        self,
        window: TimeWindow,
        prefs: Preferences,
        stored: Itinerary | None = None,
        change: Change = NO_CHANGE,
        forecast: WeatherReport | None = None,
    ) -> PlanResult | NeedInfoResult:
        """Plan the window: a new plan, or the stored one changed.

        Args:
            window: The time window.
            prefs: The preferences.
            stored: The stored plan, if any.
            change: What the message asks.
            forecast: A forecast whose rain the plan avoids.

        Returns:
            The validated plan, or what is missing to change the stored one.
        """
        if change.replace and stored is None:
            return need_info(ITINERARY_MISSING)
        if stored is not None and not stored.has_every_stop(change.replace):
            return need_info(f"which stop to change (1-{len(stored.stops)})")

        if forecast is not None:
            change = replace(change, wet=tuple(forecast.wet_periods))

        fillers = self.places.candidates(prefs.indoor() if change.wet else prefs)  # in rain the gaps fill indoors
        selected = {poi.id for poi in self.places.candidates(prefs)} if change.recheck_preferences else set()
        rebuild = refill.refill(stored, window, fillers, self.rules, change, selected=selected)
        adaptation = Adaptation.model_validate(rebuild) if stored is not None else None
        result = self._validated(PlanResult(itinerary=rebuild.itinerary, adaptation=adaptation, forecast=forecast))

        log.info("plan: %d stop(s), %s", len(result.itinerary.stops), "valid" if result.validation.ok else "invalid")

        return result

    def check_places(self, window: TimeWindow, names: list[str]) -> PlanResult | RefusalResult:
        """Check the named places fit the window in some order.

        Args:
            window: The time window.
            names: The places as the user named them.

        Returns:
            The validated plan, or why the places do not fit and the closest alternative. A refusal for names the
            dataset lacks.
        """
        lookup = self.places.lookup(names)
        if lookup.unknown:
            return RefusalResult(reason="not_in_dataset", names=lookup.unknown)

        report = builder.check_requested(lookup.places, window, self.rules)

        return self._validated(PlanResult(feasibility=report, itinerary=report.itinerary or report.alternative))

    def _validated(self, result: PlanResult) -> PlanResult:
        if result.itinerary is not None:
            result.validation = checks.validate(result.itinerary, self.rules)
            log.info("guards: %s", [(v.code, v.severity) for v in result.validation.violations] or "ok")

        return result
