"""Itinerary nodes: a new plan (or a feasibility check of named places), and edits to the current plan."""

from langgraph.runtime import Runtime

from corga.agent.state import Update, AgentState
from corga.agent.runtime.deps import Deps
from corga.agent.nodes.results import remember


def plan(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """A new itinerary from the preferences, or a feasibility check of the places the user named."""
    itineraries, named = runtime.context.itineraries, state.parsed.requested_places
    if named:
        result = itineraries.check_places(state.window, named)
    else:
        result = itineraries.plan_itinerary(state.window, state.preferences)
    return remember(result, previous=state.itinerary)


def edit_plan(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """Replace one stop, or, without a stop number, apply newly stated preferences to the whole plan.

    Unaffected stops and the window are kept either way.
    """
    itineraries, request = runtime.context.itineraries, state.parsed
    if request.stop_index is None and request.preference_updates:
        result = itineraries.revise_itinerary(state.itinerary, state.preferences)  # preferences updated by `parse`
    else:
        result = itineraries.change_stop(state.itinerary, request.stop_index, state.preferences)
    return remember(result, previous=state.itinerary)
