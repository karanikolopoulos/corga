"""Itinerary nodes: a new plan (or a feasibility check of named places), and edits to the current plan."""

from langgraph.runtime import Runtime

from corga.agent.guide import Guide
from corga.agent.state import Update, AgentState
from corga.agent.results import Adaptation, PlanResult, RefusalResult, NeedInfoResult
from corga.planning.models import Itinerary
from corga.planning.replan import Replan
from corga.agent.runtime.deps import Deps
from corga.agent.nodes.results import planned


def plan(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """A new itinerary from the preferences, or a feasibility check of the places the user named."""
    guide = runtime.context.guide
    window = state.window
    if window is None:
        return {"result": NeedInfoResult(missing="a date plus start and end time")}
    if not state.request.requested_places:
        built = guide.build_itinerary(window, guide.select_candidates(state.preferences))
        return planned(guide, PlanResult(itinerary=built), previous=state.itinerary)

    named = guide.lookup(state.request.requested_places)
    if named.unknown:
        return {"result": RefusalResult(reason="not_in_dataset", names=named.unknown)}
    report = guide.check_requested(named.places, window)
    result = PlanResult(feasibility=report, itinerary=report.itinerary or report.alternative)
    if result.itinerary is None:
        return {"result": result}
    return planned(guide, result, previous=state.itinerary)


def edit_plan(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """Replace one stop, or, without a stop number, apply newly stated preferences to the whole plan.

    Unaffected stops and the window are kept either way.
    """
    guide = runtime.context.guide
    itinerary, index = state.itinerary, state.request.stop_index  # index is 1-based
    if itinerary is None:
        return {"result": NeedInfoResult(missing="an existing itinerary to edit")}
    if index is None and state.request.preference_updates:
        replan = guide.replan_for_preferences(itinerary, state.preferences)  # preferences already updated by `parse`
        return _replanned(guide, replan, previous=itinerary)
    if index is None or not 1 <= index <= len(itinerary.stops):
        return {"result": NeedInfoResult(missing=f"which stop to change (1-{len(itinerary.stops)})")}

    replan = guide.replace_stop(itinerary, index, state.preferences)
    return _replanned(guide, replan, previous=itinerary)


def _replanned(guide: Guide, replan: Replan, previous: Itinerary) -> Update:
    """An edited plan (or the unchanged one, with the note saying why) as the turn's result."""
    result = PlanResult(itinerary=replan.itinerary, adaptation=Adaptation.model_validate(replan))
    return planned(guide, result, previous=previous)
