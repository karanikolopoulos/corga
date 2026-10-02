from abc import ABC, abstractmethod
from typing import NamedTuple
from datetime import date, timedelta
from functools import cached_property
from collections import Counter
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

from corga.domain.places import POI
from corga.domain.timeutil import Interval
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.models import Stop, Severity, Itinerary, Violation, TimeWindow, ViolationCode


class Clause(NamedTuple):
    ok: bool
    code: ViolationCode
    msg: str
    severity: Severity = "error"


class Guards(ABC):
    """A group of clauses about one part of an itinerary."""

    @abstractmethod
    def __iter__(self) -> Iterator[Clause]: ...

    @property
    def passes(self) -> bool:
        """Every clause holds, warnings included."""
        return all(clause.ok for clause in self)

    def failures(self, stop: int | None = None, poi_id: str | None = None) -> list[Violation]:
        """List the failed clauses as violations tagged with their stop and place."""
        return [
            Violation(code=c.code, severity=c.severity, stop=stop, poi_id=poi_id, message=c.msg)
            for c in self
            if not c.ok
        ]


@dataclass(frozen=True)
class ItineraryGuards(Guards):
    itinerary: Itinerary

    @property
    def has_stops(self) -> Clause:
        return Clause(
            ok=bool(self.itinerary.stops),
            code="EMPTY",
            msg="No stop fits the requested window and constraints.",
        )

    @property
    def unique_stops(self) -> Clause:
        visits = Counter(stop.poi_id for stop in self.itinerary.stops)
        names = {stop.poi_id: stop.name for stop in self.itinerary.stops}
        repeated = [names[poi_id] for poi_id, count in visits.items() if count > 1]  # each place once

        return Clause(
            ok=not repeated,
            code="DUPLICATE",
            msg=f"Repeated stops: {', '.join(repeated)}.",
        )

    @property
    def fills_window(self) -> Clause:
        """The stops, walks included, reach at least half the window."""
        window, used = self.itinerary.window, self.itinerary.minutes_used

        return Clause(
            ok=not self.itinerary.stops or self.itinerary.half_filled,
            code="SHORT_PLAN",
            msg=f"The stops fill {used} of the {window.minutes} minutes: few places fit the preferences and hours.",
            severity="warning",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.has_stops
        yield self.unique_stops
        yield self.fills_window


@dataclass(frozen=True)
class StopGuards(Guards):
    stop: Stop
    window: TimeWindow
    rules: PlanningRules

    @property
    def within_window(self) -> Clause:
        """The visit, from arrival, falls inside the window."""
        stop, window = self.stop, self.window

        return Clause(
            ok=window.span.contains(Interval(stop.arrive, stop.end)),  # from arrival: waiting for opening counts too
            code="OUTSIDE_WINDOW",
            msg=f"{stop.name} ({stop.start:%H:%M}-{stop.end:%H:%M}) falls outside "
            f"{window.start:%H:%M}-{window.end:%H:%M}.",
        )

    @property
    def full_visit(self) -> Clause:
        """The visit lasts the place's suggested time."""
        minutes = self.stop.poi.suggested_visit_minutes

        return Clause(
            ok=self.stop.end - self.stop.start >= timedelta(minutes=minutes),
            code="VISIT_TOO_SHORT",
            msg=f"{self.stop.name} needs about {minutes} min.",
        )

    @property
    def open_during_visit(self) -> Clause:
        """The place is open for the whole visit. Unknown hours fail."""
        stop, hours = self.stop, self.stop.poi.opening_hours
        if hours is None:
            return Clause(
                ok=False,
                code="UNKNOWN_HOURS",
                msg=f"No confirmed opening hours for {stop.name}; access cannot be promised.",
            )

        return Clause(
            ok=stop.span in hours,
            code="CLOSED",
            msg=f"{stop.name} is not open for the whole visit {stop.start:%H:%M}-{stop.end:%H:%M} on {stop.start:%A}.",
        )

    @property
    def short_wait(self) -> Clause:
        """The wait for opening stays within the limit."""
        wait = self.stop.start - self.stop.arrive

        return Clause(
            ok=wait <= timedelta(minutes=self.rules.max_wait_minutes),
            code="LONG_WAIT",
            msg=f"Waiting {wait} for {self.stop.name} to open.",
            severity="warning",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.within_window
        yield self.full_visit
        yield self.open_during_visit
        yield self.short_wait


@dataclass(frozen=True)
class LegGuards(Guards):
    prev_stop: Stop
    stop: Stop
    rules: PlanningRules

    @cached_property
    def walk_minutes(self) -> int:
        return self.rules.walking.minutes(self.prev_stop.poi.location, self.stop.poi.location)

    @property
    def enough_travel_time(self) -> Clause:
        return Clause(
            ok=self.stop.arrive - self.prev_stop.end >= timedelta(minutes=self.walk_minutes),
            code="TRAVEL_TOO_SHORT",
            msg=f"Walking from {self.prev_stop.name} to {self.stop.name} takes about {self.walk_minutes} min.",
        )

    @property
    def walkable(self) -> Clause:
        """The walk stays within the longest walk allowed."""
        limit = self.rules.max_leg_minutes

        return Clause(
            ok=self.walk_minutes <= limit,
            code="LEG_TOO_LONG",
            msg=f"{self.stop.name} is ~{self.walk_minutes} min on foot from {self.prev_stop.name} (limit {limit} min).",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.enough_travel_time
        yield self.walkable


@dataclass(frozen=True)
class VisitGuards(Guards):
    """One visit: the stop itself and, after the first stop, the walk into it."""

    prev_stop: Stop | None  # None for the first stop of the day
    stop: Stop
    window: TimeWindow
    rules: PlanningRules

    def __iter__(self) -> Iterator[Clause]:
        # the stop: in the window, a full visit, open throughout, no long wait
        yield from StopGuards(self.stop, self.window, self.rules)

        # the first stop has no walk leading to it
        if self.prev_stop is None:
            return

        # the walk from the previous stop: enough time to cover it, and short enough
        yield from LegGuards(self.prev_stop, self.stop, self.rules)


def visit_minutes(pois: Sequence[POI]) -> int:
    """Add up the visits' minutes, walking excluded."""
    return sum(poi.suggested_visit_minutes for poi in pois)


@dataclass(frozen=True)
class RequestGuards(Guards):
    """A request to visit specific places, taken as a whole."""

    pois: Sequence[POI]
    window: TimeWindow

    @property
    def fits_in_window(self) -> Clause:
        """The visits alone fit in the window."""
        visit = visit_minutes(self.pois)

        return Clause(
            ok=visit <= self.window.minutes,
            code="TIME_BUDGET",
            msg=f"The visits alone need {visit} min but the window is {self.window.minutes} min.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.fits_in_window


@dataclass(frozen=True)
class PlaceGuards(Guards):
    """One requested place, on the requested day, among the other requested places."""

    poi: POI
    others: Sequence[POI]
    day: date
    rules: PlanningRules

    @property
    def open_that_day(self) -> Clause:
        """The place is open that day. Unknown hours fail."""
        poi, hours = self.poi, self.poi.opening_hours
        if hours is None:
            return Clause(ok=False, code="UNKNOWN_HOURS", msg=f"{poi.name} has no confirmed opening hours.")

        return Clause(
            ok=poi.is_open_on(self.day),
            code="CLOSED_DAY",
            msg=f"{poi.name} is closed on {self.day:%A} (fixture hours).",
        )

    @cached_property
    def nearest_walk(self) -> int:
        """Minutes on foot to the closest other requested place (0 if there is none)."""
        walks = (self.rules.walking.minutes(self.poi.location, other.location) for other in self.others)

        return min(walks, default=0)

    @property
    def within_walking_distance(self) -> Clause:
        return Clause(
            ok=self.nearest_walk <= self.rules.max_leg_minutes,
            code="LEG_TOO_LONG",
            msg=f"{self.poi.name} is ~{self.nearest_walk} min on foot from the nearest other requested place.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.open_that_day
        yield self.within_walking_distance
