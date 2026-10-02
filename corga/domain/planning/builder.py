import logging

from typing import NamedTuple
from operator import attrgetter
from itertools import permutations
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, computed_field

from corga.domain.places import POI
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.checks import validate
from corga.domain.planning.guards import PlaceGuards, VisitGuards, RequestGuards, visit_minutes
from corga.domain.planning.models import Itinerary, Violation, TimeWindow
from corga.domain.planning.schedule import Schedule

log = logging.getLogger(__name__)

MAX_PERMUTED = 6  # requested places up to this count are tried in every order


class Option(NamedTuple):
    poi: POI
    schedule: Schedule  # the plan with this candidate visited next
    cost: float  # lower is better


class FeasibilityReport(BaseModel):
    """Whether named places fit the window, why not, and an alternative."""

    available_minutes: int
    visit_minutes: int
    issues: list[Violation] = []
    itinerary: Itinerary | None = None  # a visiting order that passes validation
    alternative: Itinerary | None = None

    @computed_field
    @property
    def feasible(self) -> bool:
        return self.itinerary is not None


def _itinerary(window: TimeWindow, schedule: Schedule, rules: PlanningRules) -> Itinerary:
    return Itinerary(window=window, stops=list(schedule), assumptions=rules.assumptions())


@dataclass(frozen=True)
class ItineraryBuilder:
    """Builds a plan greedily: the cheapest next stop that fits, until none does."""

    window: TimeWindow
    rules: PlanningRules

    def build(self, candidates: Sequence[POI], prefix: Sequence[POI] = ()) -> Itinerary:
        """Keep `prefix`, then add the best next stop until nothing fits.

        Args:
            candidates: The places to choose from, best first.
            prefix: The places to visit first, in order.

        Returns:
            The itinerary.
        """
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
        """Find the cheapest candidate to visit next.

        Args:
            plan: The schedule so far.
            remaining: The candidates left, best first.

        Returns:
            The cheapest option that fits, or None.
        """
        options = [self.option(plan, rank, poi) for rank, poi in enumerate(remaining)]
        fitting = [o for o in options if self.fits(plan, o.schedule)]
        best = min(fitting, key=attrgetter("cost"), default=None)  # ties go to the better-ranked candidate

        after = plan.here.id if plan.here else "the start"
        log.debug(
            "next after %s: %d of %d fit, best %s", after, len(fitting), len(options), best.poi.id if best else None
        )

        return best

    def option(self, plan: Schedule, rank: int, poi: POI) -> Option:
        """Try `poi` as the next stop.

        Args:
            plan: The schedule so far.
            rank: The place's position among the candidates.
            poi: The place.

        Returns:
            The option: its rank plus a penalty for the walk to it as the cost.
        """
        trial = plan.then(poi)
        cost = rank + self.rules.walk_penalty * trial.stops[-1].walk_minutes

        return Option(poi=poi, schedule=trial, cost=cost)

    def fits(self, base: Schedule, trial: Schedule) -> bool:
        """Check the visit `trial` adds to `base` can stay.

        Args:
            base: The schedule before the visit.
            trial: The schedule with it.

        Returns:
            Whether the stop itself and the walk into it both pass.
        """
        prev_stop = base.stops[-1] if base.stops else None

        return VisitGuards(prev_stop, trial.stops[-1], self.window, self.rules).passes


def build_itinerary(
    window: TimeWindow, candidates: Sequence[POI], rules: PlanningRules, prefix: Sequence[POI] = ()
) -> Itinerary:
    return ItineraryBuilder(window, rules).build(candidates, prefix)


def _first_valid(orders: Iterable[Sequence[POI]], window: TimeWindow, rules: PlanningRules) -> Itinerary | None:
    """Return the first visiting order that passes validation, or None."""
    for order in orders:
        itinerary = _itinerary(window, Schedule.of(order, window.start, rules.walking), rules)
        if validate(itinerary, rules).ok:
            return itinerary

    return None


def _reasons(pois: Sequence[POI], window: TimeWindow, rules: PlanningRules) -> list[Violation]:
    """Explain why no visiting order works."""
    reasons = RequestGuards(pois, window).failures()

    for poi in pois:
        others = [other for other in pois if other.id != poi.id]
        reasons += PlaceGuards(poi, others, window.day, rules).failures(poi_id=poi.id)

    return reasons


def check_requested(pois: Sequence[POI], window: TimeWindow, rules: PlanningRules) -> FeasibilityReport:
    """Check the named places fit the window in some order.

    Every order is tried for up to six places.

    Args:
        pois: The named places.
        window: The time window.
        rules: The planning rules.

    Returns:
        The report: a fitting order, or why none fits and the best plan from the places open that day.
    """
    visit = visit_minutes(pois)

    # feasible if any visiting order passes validation (all orders for small requests)
    orders = permutations(pois) if len(pois) <= MAX_PERMUTED else [tuple(pois)]
    itinerary = _first_valid(orders, window, rules)
    if itinerary is not None:
        return FeasibilityReport(available_minutes=window.minutes, visit_minutes=visit, itinerary=itinerary)

    # otherwise explain why, and offer a plan from the places open that day
    open_that_day = [p for p in pois if p.is_open_on(window.day)]
    alternative = build_itinerary(window, open_that_day, rules)

    return FeasibilityReport(
        available_minutes=window.minutes,
        visit_minutes=visit,
        issues=_reasons(pois, window, rules),
        alternative=alternative if alternative.stops else None,
    )
