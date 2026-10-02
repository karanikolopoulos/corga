"""Producing a plan: build one greedily, replace one stop, or try requested places in every order."""

from typing import NamedTuple
from operator import attrgetter
from itertools import permutations
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, computed_field

from corga.data.models import POI
from corga.planning.rules import PlanningRules
from corga.planning.checks import validate
from corga.planning.guards import PlaceGuards, VisitGuards, RequestGuards, visit_minutes
from corga.planning.models import Itinerary, Violation, TimeWindow
from corga.planning.schedule import Schedule

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

        return _itinerary(self.window, plan, self.rules)

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
        prev_stop = base.stops[-1] if base.stops else None
        return VisitGuards(prev_stop, trial.stops[-1], self.window, self.rules).passes


def build_itinerary(
    window: TimeWindow, candidates: Sequence[POI], rules: PlanningRules, prefix: Sequence[POI] = ()
) -> Itinerary:
    return ItineraryBuilder(window, rules).build(candidates, prefix)


def _itinerary(window: TimeWindow, schedule: Schedule, rules: PlanningRules) -> Itinerary:
    """The schedule's stops as an itinerary for the window, with the rules' assumptions."""
    return Itinerary(window=window, stops=list(schedule), assumptions=rules.assumptions())


def _first_valid(orders: Iterable[Sequence[POI]], window: TimeWindow, rules: PlanningRules) -> Itinerary | None:
    """The itinerary of the first visiting order that passes validation, or None if none does."""
    for order in orders:
        itinerary = _itinerary(window, Schedule.of(order, window.start, rules.walking), rules)
        if validate(itinerary, rules).ok:
            return itinerary
    return None


def replace_stop(
    itinerary: Itinerary,
    index: int,
    candidates: Sequence[POI],
    rules: PlanningRules,
) -> Itinerary | None:
    """Swap stop `index` (1-based) for the first candidate that keeps the whole plan valid."""
    pois = itinerary.pois
    if not 1 <= index <= len(pois):
        raise ValueError(f"stop index {index} out of range 1..{len(pois)}")
    taken = set(itinerary.poi_ids)
    before, after = pois[: index - 1], pois[index:]
    orders = ([*before, cand, *after] for cand in candidates if cand.id not in taken)
    return _first_valid(orders, itinerary.window, rules)


class FeasibilityReport(BaseModel):
    available_minutes: int
    visit_minutes: int
    issues: list[Violation] = []
    itinerary: Itinerary | None = None  # a visiting order that passes validation
    alternative: Itinerary | None = None

    @computed_field
    @property
    def feasible(self) -> bool:
        return self.itinerary is not None


def _reasons(pois: Sequence[POI], window: TimeWindow, rules: PlanningRules) -> list[Violation]:
    """Why no visiting order works: the request as a whole, then each place in it."""
    reasons = RequestGuards(pois, window).failures()
    for poi in pois:
        others = [other for other in pois if other.id != poi.id]
        reasons += PlaceGuards(poi, others, window.day, rules).failures(poi_id=poi.id)
    return reasons


def check_requested(pois: Sequence[POI], window: TimeWindow, rules: PlanningRules) -> FeasibilityReport:
    visit = visit_minutes(pois)

    # Feasible if any visiting order passes validation (all orders for small requests).
    orders = permutations(pois) if len(pois) <= MAX_PERMUTED else [tuple(pois)]
    itinerary = _first_valid(orders, window, rules)
    if itinerary is not None:
        return FeasibilityReport(available_minutes=window.minutes, visit_minutes=visit, itinerary=itinerary)

    # Otherwise explain why, and offer the best plan built from the places open that day.
    open_that_day = [p for p in pois if p.is_open_on(window.day)]
    alternative = build_itinerary(window, open_that_day, rules)
    return FeasibilityReport(
        available_minutes=window.minutes,
        visit_minutes=visit,
        issues=_reasons(pois, window, rules),
        alternative=alternative if alternative.stops else None,
    )
