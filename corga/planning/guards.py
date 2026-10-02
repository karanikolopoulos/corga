"""Named constraint checks, each pairing a condition with the message shown when it fails."""

from abc import ABC, abstractmethod
from typing import NamedTuple
from datetime import date, timedelta
from functools import cached_property
from collections import Counter
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

from corga.data.models import POI
from corga.planning.rules import PlanningRules
from corga.planning.models import Stop, Severity, Itinerary, Violation, TimeWindow


class Clause(NamedTuple):
    ok: bool
    code: str
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
        """Every clause that does not hold, as a violation tagged with the stop and place it belongs to."""
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
        counts = Counter(stop.name for stop in self.itinerary.stops)
        repeated = [name for name, count in counts.items() if count > 1]
        return Clause(
            ok=not repeated,
            code="DUPLICATE",
            msg=f"Repeated stops: {', '.join(repeated)}.",
        )

    def __iter__(self) -> Iterator[Clause]:
        yield self.has_stops
        yield self.unique_stops


@dataclass(frozen=True)
class StopGuards(Guards):
    stop: Stop
    poi: POI
    window: TimeWindow
    rules: PlanningRules

    @property
    def within_window(self) -> Clause:
        stop, window = self.stop, self.window
        return Clause(
            ok=window.start <= stop.arrive and stop.end <= window.end,
            code="OUTSIDE_WINDOW",
            msg=f"{stop.name} ({stop.start:%H:%M}-{stop.end:%H:%M}) falls outside "
            f"{window.start:%H:%M}-{window.end:%H:%M}.",
        )

    @property
    def full_visit(self) -> Clause:
        minutes = self.poi.suggested_visit_minutes
        return Clause(
            ok=self.stop.end - self.stop.start >= timedelta(minutes=minutes),
            code="VISIT_TOO_SHORT",
            msg=f"{self.stop.name} needs about {minutes} min.",
        )

    @property
    def open_during_visit(self) -> Clause:
        stop, hours = self.stop, self.poi.opening_hours
        if hours is None:
            return Clause(
                ok=False,
                code="UNKNOWN_HOURS",
                msg=f"No confirmed opening hours for {stop.name}; access cannot be promised.",
            )
        return Clause(
            ok=(stop.start, stop.end) in hours,
            code="CLOSED",
            msg=f"{stop.name} is not open for the whole visit {stop.start:%H:%M}-{stop.end:%H:%M} on {stop.start:%A}.",
        )

    @property
    def short_wait(self) -> Clause:
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
    prev_poi: POI
    stop: Stop
    poi: POI
    rules: PlanningRules

    @cached_property
    def walk_minutes(self) -> int:
        return self.rules.walking.minutes(self.prev_poi.location, self.poi.location)

    @property
    def enough_travel_time(self) -> Clause:
        return Clause(
            ok=self.stop.arrive - self.prev_stop.end >= timedelta(minutes=self.walk_minutes),
            code="TRAVEL_TOO_SHORT",
            msg=f"Walking from {self.prev_stop.name} to {self.stop.name} takes about {self.walk_minutes} min.",
        )

    @property
    def walkable(self) -> Clause:
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
class RequestGuards(Guards):
    """A request to visit specific places, taken as a whole."""

    pois: Sequence[POI]
    window: TimeWindow

    @property
    def fits_in_window(self) -> Clause:
        visit = sum(poi.suggested_visit_minutes for poi in self.pois)
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
        poi, hours = self.poi, self.poi.opening_hours
        if hours is None:
            return Clause(ok=False, code="UNKNOWN_HOURS", msg=f"{poi.name} has no confirmed opening hours.")
        return Clause(
            ok=hours.is_open_on(self.day),
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
