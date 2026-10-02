"""Changing an existing plan. Every change returns a `Replan`: the new itinerary, which stops it replaced, and a note.

Rebuilding: keep the stops before the first one that no longer fits and rebuild the rest of the window. Two reasons
a stop no longer fits: the updated preferences do not select it, or it is outdoors while it rains.
Replacing one stop the user named: swap it for a place matching the preferences, keeping the others.
"""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, computed_field

from corga.data.models import POI
from corga.data.timeutil import Interval
from corga.planning.rules import PlanningRules
from corga.planning.models import Stop, Itinerary, Preferences
from corga.planning.builder import replace_stop, build_itinerary
from corga.planning.candidates import Ranker, select_candidates

FALLBACK_NOTE = " No place matching the interests fits; it matches the other preferences."


class Replan(BaseModel):
    """An itinerary after a change, the places it no longer visits, and a note saying what happened."""

    replaced: list[str] = []
    itinerary: Itinerary
    note: str

    @computed_field
    @property
    def changed(self) -> bool:
        return bool(self.replaced)


def replaced_ids(old: Itinerary, new: Itinerary) -> list[str]:
    """The places `old` visits and `new` does not."""
    return [pid for pid in old.poi_ids if pid not in new.poi_ids]


def rebuild_from_first_misfit(
    itinerary: Itinerary,
    fits: Callable[[Stop], bool],
    candidates: Callable[[], Sequence[POI]],
    rules: PlanningRules,
) -> Itinerary:
    """Keep the stops before the first one that fails `fits`, then fill the rest of the window from `candidates`.

    `candidates` is called only when a stop fails; the window is unchanged, and so is the itinerary if every stop fits.
    """
    first = next((i for i, stop in enumerate(itinerary.stops) if not fits(stop)), None)
    if first is None:
        return itinerary
    return build_itinerary(itinerary.window, candidates(), rules, prefix=itinerary.pois[:first])


def adapt_for_preferences(itinerary: Itinerary, candidates: Sequence[POI], rules: PlanningRules) -> Replan:
    """Rebuild from the first stop that is not among `candidates` (the places the preferences now select)."""
    selected = {poi.id for poi in candidates}
    new = rebuild_from_first_misfit(itinerary, lambda stop: stop.poi_id in selected, lambda: candidates, rules)
    replaced = replaced_ids(itinerary, new)
    if not replaced:
        return Replan(itinerary=new, note="Every stop still matches the preferences.")
    note = f"Replaced {len(replaced)} stop(s) that no longer match the preferences."
    return Replan(replaced=replaced, itinerary=new, note=note)


def is_exposed(stop: Stop, periods: Sequence[Interval]) -> bool:
    """An outdoor or mixed stop (mixed is not shelter) visited while it rains."""
    return stop.setting in ("outdoor", "mixed") and any(period.overlaps(stop.span) for period in periods)


def adapt_for_weather(
    itinerary: Itinerary,
    periods: Sequence[Interval],
    indoor_candidates: Callable[[], Sequence[POI]],
    rules: PlanningRules,
) -> Replan:
    """The itinerary rebuilt from indoor places from its first rain-exposed stop onwards."""
    new = rebuild_from_first_misfit(itinerary, lambda stop: not is_exposed(stop, periods), indoor_candidates, rules)
    replaced = replaced_ids(itinerary, new)
    if not replaced:
        return Replan(itinerary=new, note="No outdoor or mixed stop overlaps forecast rain.")
    note = f"Rain expected from {periods[0].start:%H:%M}; replaced {len(replaced)} exposed stop(s) with indoor places."
    return Replan(replaced=replaced, itinerary=new, note=note)


def replace_stop_for_preferences(
    itinerary: Itinerary,
    index: int,
    pois: Sequence[POI],
    prefs: Preferences,
    *,
    rules: PlanningRules,
    ranker: Ranker | None = None,
) -> Replan:
    """Replace stop `index` (1-based) with a place matching `prefs`, or, if none fits, with one that matches every
    preference except the interests; the itinerary is unchanged if neither fits."""
    for attempt, fallback in ((prefs, ""), (prefs.without_interests(), FALLBACK_NOTE)):
        new = replace_stop(itinerary, index, select_candidates(pois, attempt, ranker), rules)
        if new is None:
            continue
        old, chosen = itinerary.stops[index - 1], new.stops[index - 1]
        note = f"Replaced stop {index}, {old.name}, with {chosen.name}.{fallback}"
        return Replan(replaced=[old.poi_id], itinerary=new, note=note)

    old = itinerary.stops[index - 1]
    note = f"No alternative for stop {index}, {old.name}, fits the window and constraints."
    return Replan(itinerary=itinerary, note=note)
