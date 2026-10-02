"""Checking the itinerary a result carries, and keeping it in state only if it is valid."""

from corga.agent.guide import Guide
from corga.agent.state import Update
from corga.agent.results import PlanResult, WeatherResult
from corga.planning.models import Itinerary


def planned(guide: Guide, result: PlanResult | WeatherResult, previous: Itinerary | None) -> Update:
    """Validate the result's itinerary and keep it in state only if it is valid (else `previous`)."""
    itinerary = result.itinerary
    result.validation = guide.validate(itinerary)
    kept = itinerary if result.validation.ok else previous
    return {"itinerary": kept, "result": result}
