"""Deterministic hard-constraint checks on an itinerary."""

from corga.planning.rules import PlanningRules
from corga.planning.guards import VisitGuards, ItineraryGuards
from corga.planning.models import Itinerary, ValidationResult


def validate(itinerary: Itinerary, rules: PlanningRules) -> ValidationResult:
    # Whole itinerary: at least one stop, no repeated places.
    violations = ItineraryGuards(itinerary).failures()

    # Each visit: the stop itself, then the walk into it; both reported on that stop.
    prev_stop = None
    for n, stop in enumerate(itinerary.stops, start=1):
        violations += VisitGuards(prev_stop, stop, itinerary.window, rules).failures(stop=n, poi_id=stop.poi_id)
        prev_stop = stop

    # All failures are kept; only errors make the itinerary invalid.
    return ValidationResult(violations=violations)
