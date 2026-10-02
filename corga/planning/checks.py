"""Deterministic hard-constraint checks. The LLM never decides feasibility."""

from itertools import pairwise

from corga.data.store import POIStore
from corga.planning.rules import PlanningRules
from corga.planning.guards import LegGuards, StopGuards, ItineraryGuards
from corga.planning.models import Itinerary, ValidationResult


def validate(itinerary: Itinerary, store: POIStore, rules: PlanningRules) -> ValidationResult:
    visits = [(stop, store.get(stop.poi_id)) for stop in itinerary.stops]  # each stop with its POI record

    # Whole itinerary: at least one stop, no repeated places.
    violations = ItineraryGuards(itinerary).failures()

    # Each stop: inside the window, full visit, open for the whole visit, no long wait.
    for n, (stop, poi) in enumerate(visits, start=1):
        violations += StopGuards(stop, poi, itinerary.window, rules).failures(stop=n, poi_id=poi.id)

    # Each leg between consecutive stops, reported on the stop it leads to (so numbering starts at 2).
    for n, ((prev_stop, prev_poi), (stop, poi)) in enumerate(pairwise(visits), start=2):
        violations += LegGuards(prev_stop, prev_poi, stop, poi, rules).failures(stop=n, poi_id=poi.id)

    # Every failure is kept, so an infeasible plan can explain all of its reasons; warnings do not make it invalid.
    return ValidationResult(violations=violations)
