from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.guards import VisitGuards, ItineraryGuards
from corga.domain.planning.models import Itinerary, ValidationResult


def validate(itinerary: Itinerary, rules: PlanningRules) -> ValidationResult:
    """Check every hard constraint of the itinerary.

    Args:
        itinerary: The itinerary.
        rules: The planning rules.

    Returns:
        Every failure. Only errors make the itinerary invalid.
    """
    # whole itinerary: at least one stop, no repeated places
    violations = ItineraryGuards(itinerary).failures()

    # each visit: the stop itself, then the walk into it, both reported on that stop
    prev_stop = None

    for n, stop in enumerate(itinerary.stops, start=1):
        violations += VisitGuards(prev_stop, stop, itinerary.window, rules).failures(stop=n, poi_id=stop.poi_id)
        prev_stop = stop

    # all failures are kept, only errors make the itinerary invalid
    return ValidationResult(violations=violations)
