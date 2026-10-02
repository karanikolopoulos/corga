"""Everything that produces a plan: build one greedily, replace one stop, rebuild from the first stop that no longer
fits, or try requested places in every order."""

from typing import NamedTuple
from operator import attrgetter
from itertools import permutations
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from corga.data.store import POIStore
from corga.data.models import POI
from corga.planning.rules import PlanningRules
from corga.planning.checks import validate
from corga.planning.guards import LegGuards, StopGuards, PlaceGuards, RequestGuards
from corga.planning.models import Stop, Itinerary, Violation, TimeWindow, Preferences
from corga.planning.schedule import Schedule
from corga.planning.candidates import Ranker, select_candidates

MAX_PERMUTED = 6  # requested places up to this count are tried in every order


class Option(NamedTuple):
    """One candidate tried as the next stop."""

    poi: POI
    schedule: Schedule  # the plan with this candidate visited next
    cost: float  # lower is better


@dataclass(frozen=True)
class ItineraryBuilder:
    window: TimeWindow
    rules: PlanningRules

    def build(self, candidates: Sequence[POI], prefix: Sequence[POI] = ()) -> Itinerary:
        """Keep `prefix` as given, then add the best next stop until the day is full or nothing fits."""
        plan = Schedule.of(prefix, self.window.start, self.rules.walking)
        taken = {p.id for p in prefix}
        remaining = [c for c in candidates if c.id not in taken]  # keeps the ranked order

        while len(plan.stops) < self.rules.max_stops:
            best = self.best_next(plan, remaining)
            if best is None:
                break  # nothing else fits the window
            plan = best.schedule
            remaining.remove(best.poi)

        return Itinerary(window=self.window, stops=list(plan), assumptions=self.rules.assumptions())

    def best_next(self, plan: Schedule, remaining: Sequence[POI]) -> Option | None:
        """The cheapest candidate that can be visited next, or None if none fits."""
        options = [self.option(plan, rank, poi) for rank, poi in enumerate(remaining)]
        fitting = [o for o in options if self.fits(plan, o.schedule)]
        return min(fitting, key=attrgetter("cost"), default=None)  # ties go to the better-ranked candidate

    def option(self, plan: Schedule, rank: int, poi: POI) -> Option:
        """`poi` tried as the next stop; the cost is its place in the ranking plus a penalty for the walk to it."""
        trial = plan.then(poi)
        cost = rank + self.rules.walk_penalty * trial.stops[-1].walk_minutes
        return Option(poi=poi, schedule=trial, cost=cost)

    def fits(self, base: Schedule, trial: Schedule) -> bool:
        """Whether the visit `trial` adds to `base` can stay: the stop itself and the walk into it both pass."""
        stop, poi = trial.stops[-1], trial.here  # the candidate visit just added

        # The stop: inside the window, open for the whole visit, no long wait.
        if not StopGuards(stop, poi, self.window, self.rules).passes:
            return False

        if base.at_start:  # first stop: there is no walk leading to it
            return True

        # The walk from the previous stop: short enough, and enough time to cover it.
        return LegGuards(base.stops[-1], base.here, stop, poi, self.rules).passes


def build_itinerary(
    window: TimeWindow, candidates: Sequence[POI], rules: PlanningRules, prefix: Sequence[POI] = ()
) -> Itinerary:
    return ItineraryBuilder(window, rules).build(candidates, prefix)


def replace_stop(
    itinerary: Itinerary,
    index: int,
    candidates: Sequence[POI],
    store: POIStore,
    rules: PlanningRules,
) -> Itinerary | None:
    """Swap stop `index` (1-based) for the first candidate that keeps the whole plan valid."""
    pois = [store.get(pid) for pid in itinerary.poi_ids]
    if not 1 <= index <= len(pois):
        raise ValueError(f"stop index {index} out of range 1..{len(pois)}")
    taken = set(itinerary.poi_ids)
    for cand in candidates:
        if cand.id in taken:
            continue
        order = [*pois[: index - 1], cand, *pois[index:]]
        stops = list(Schedule.of(order, itinerary.window.start, rules.walking))
        new = itinerary.model_copy(update={"stops": stops})
        if validate(new, store, rules).ok:
            return new
    return None


def replace_stop_for_preferences(
    itinerary: Itinerary,
    index: int,
    pois: Sequence[POI],
    prefs: Preferences,
    *,
    store: POIStore,
    rules: PlanningRules,
    ranker: Ranker | None = None,
) -> Itinerary | None:
    """Replace stop `index` (1-based) with a place matching `prefs`, or, if none fits, with one that matches every
    preference except the interests; None if neither fits."""
    for attempt in (prefs, prefs.without_interests()):
        new = replace_stop(itinerary, index, select_candidates(pois, attempt, ranker), store, rules)
        if new is not None:
            return new
    return None


class Replan(BaseModel):
    """An itinerary rebuilt from its first stop that no longer fits, and which stops it replaced."""

    changed: bool
    replaced: list[str] = []
    itinerary: Itinerary
    note: str = ""


def replan_from_first_misfit(
    itinerary: Itinerary,
    fits: Callable[[Stop], bool],
    candidates: Callable[[], Sequence[POI]],
    store: POIStore,
    rules: PlanningRules,
) -> Replan:
    """Keep the stops before the first one that fails `fits`, then fill the rest of the window from `candidates`.

    `candidates` is called only when a stop fails; the window is unchanged.
    """
    first = next((i for i, stop in enumerate(itinerary.stops) if not fits(stop)), None)
    if first is None:
        return Replan(changed=False, itinerary=itinerary)
    kept = [store.get(stop.poi_id) for stop in itinerary.stops[:first]]
    new = build_itinerary(itinerary.window, candidates(), rules, prefix=kept)
    replaced = [pid for pid in itinerary.poi_ids if pid not in new.poi_ids]
    return Replan(changed=True, replaced=replaced, itinerary=new)


class FeasibilityReport(BaseModel):
    feasible: bool
    available_minutes: int
    visit_minutes: int
    issues: list[Violation] = []
    itinerary: Itinerary | None = None
    alternative: Itinerary | None = None


def _reasons(pois: Sequence[POI], window: TimeWindow, rules: PlanningRules) -> list[Violation]:
    """Why no visiting order works: the request as a whole, then each place in it."""
    reasons = RequestGuards(pois, window).failures()
    for poi in pois:
        others = [other for other in pois if other.id != poi.id]
        reasons += PlaceGuards(poi, others, window.start.date(), rules).failures(poi_id=poi.id)
    return reasons


def check_requested(
    pois: Sequence[POI], window: TimeWindow, store: POIStore, rules: PlanningRules
) -> FeasibilityReport:
    visit = sum(p.suggested_visit_minutes for p in pois)

    # Feasible if any visiting order passes validation (all orders for small requests).
    orders = permutations(pois) if len(pois) <= MAX_PERMUTED else [tuple(pois)]
    for order in orders:
        stops = list(Schedule.of(order, window.start, rules.walking))
        it = Itinerary(window=window, stops=stops, assumptions=rules.assumptions())
        if validate(it, store, rules).ok:
            return FeasibilityReport(feasible=True, available_minutes=window.minutes, visit_minutes=visit, itinerary=it)

    # Otherwise explain why, and offer the best plan built from the places open that day.
    day = window.start.date()
    open_that_day = [p for p in pois if p.is_open_on(day)]
    alternative = build_itinerary(window, open_that_day, rules)
    return FeasibilityReport(
        feasible=False,
        available_minutes=window.minutes,
        visit_minutes=visit,
        issues=_reasons(pois, window, rules),
        alternative=alternative if alternative.stops else None,
    )
