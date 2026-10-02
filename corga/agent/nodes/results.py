"""The node update for a turn's result: the result, and the plan the conversation keeps."""

from corga.agent.state import Update
from corga.services.results import Result, PlannedResult
from corga.domain.planning.models import Itinerary


def remember(result: Result, previous: Itinerary | None) -> Update:
    """The result and, when it carries an itinerary, that itinerary if it passed validation, else `previous`."""
    if not isinstance(result, PlannedResult) or result.itinerary is None:
        return {"result": result}
    kept = result.itinerary if result.validation.ok else previous
    return {"itinerary": kept, "result": result}
