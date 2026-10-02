import logging

from typing import Literal
from dataclasses import field, dataclass
from collections.abc import Sequence

from pydantic import BaseModel, computed_field

from corga.domain.places import POI
from corga.domain.timeutil import Interval
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.guards import LegGuards
from corga.domain.planning.models import Stop, Itinerary, TimeWindow
from corga.domain.planning.builder import ItineraryBuilder
from corga.domain.planning.schedule import Schedule

log = logging.getLogger(__name__)

RemovalReason = Literal["other day", "released", "replaced", "preferences", "rain", "window"]


@dataclass(frozen=True)
class Change:
    """What a message asks of the stored plan, resolved by code."""

    replace: frozenset[int] = frozenset()  # 1-based stop numbers to drop
    keep_others: bool = True  # False: the user lets every stored stop go
    recheck_preferences: bool = False  # drop the stops the preferences no longer select
    wet: tuple[Interval, ...] = ()  # rain periods of a forecast fetched this turn


NO_CHANGE = Change()


class Removal(BaseModel):
    poi_id: str
    name: str
    reason: RemovalReason


class Rebuild(BaseModel):
    """The plan after a change.

    The stops removed with their reason, and the places added.
    """

    itinerary: Itinerary
    removed: list[Removal] = []
    added: list[str] = []  # place names

    @computed_field
    @property
    def changed(self) -> bool:
        return bool(self.removed or self.added)


def is_exposed(stop: Stop, periods: Sequence[Interval]) -> bool:
    return stop.setting in ("outdoor", "mixed") and any(period.overlaps(stop.span) for period in periods)


@dataclass(frozen=True)
class GapBuilder(ItineraryBuilder):
    """The builder's greedy choice inside one gap.

    A place must end before `until` (a kept stop) or the window's end, out of the rain.
    """

    until: Stop | None = None
    wet: tuple[Interval, ...] = field(default=())

    def fits(self, base: Schedule, trial: Schedule) -> bool:
        """Check the visit stays out of the rain and leaves time to walk to `until`.

        Args:
            base: The schedule before the visit.
            trial: The schedule with it.

        Returns:
            Whether the visit can stay in the gap.
        """
        added = trial.stops[-1]
        if not super().fits(base, trial) or is_exposed(added, self.wet):
            return False

        if self.until is None:
            return True

        reached = self.until.model_copy(update={"arrive": self.until.start})  # as `walked_to` sets it

        return LegGuards(added, reached, self.rules).passes


def removal_reason(n: int, stop: Stop, window: TimeWindow, selected: set[str], change: Change) -> RemovalReason | None:
    """Give the reason a stored stop goes. The first that applies wins.

    Args:
        n: The stop's number (1 is the first).
        stop: The stop.
        window: The time window.
        selected: The ids the current preferences select.
        change: What the message asks.

    Returns:
        The reason, or None when the stop stays.
    """
    if stop.start.date() != window.day:
        return "other day"
    if not change.keep_others:
        return "released"
    if n in change.replace:
        return "replaced"
    if change.recheck_preferences and stop.poi_id not in selected:
        return "preferences"
    if is_exposed(stop, change.wet):
        return "rain"

    return None if window.span.contains(stop.span) else "window"


def split_stops(
    stored: Sequence[Stop], window: TimeWindow, selected: set[str], change: Change
) -> tuple[list[Stop], list[Removal]]:
    """Split the stored stops into those that stay and those that go.

    Args:
        stored: The stored stops.
        window: The time window.
        selected: The ids the current preferences select.
        change: What the message asks.

    Returns:
        The stops that stay, in order, and the removals with their reason.
    """
    kept: list[Stop] = []
    removed: list[Removal] = []

    for n, stop in enumerate(stored, start=1):
        reason = removal_reason(n, stop, window, selected, change)
        if reason is None:
            kept.append(stop)
            continue

        log.debug("stop %d, %s, removed: %s", n, stop.name, reason)
        removed.append(Removal(poi_id=stop.poi_id, name=stop.name, reason=reason))

    return kept, removed


def gap_start(prev: Stop | None, window: TimeWindow, rules: PlanningRules) -> Schedule:
    if prev is None:
        return Schedule(rules.walking, clock=window.start)

    return Schedule(rules.walking, clock=prev.end, stops=(prev,))


def fill_gap(builder: GapBuilder, start: Schedule, remaining: list[POI], room: int) -> list[Stop]:
    """Add the builder's stops after `start` while there is `room`.

    Args:
        builder: Chooses the stops for this gap.
        start: The schedule before the gap.
        remaining: The candidates left. Places used leave it.
        room: How many stops may be added.

    Returns:
        The stops added.
    """
    plan, added = start, []
    while len(added) < room:
        best = builder.best_next(plan, remaining)
        if best is None:
            break

        plan = best.schedule
        remaining.remove(best.poi)
        added.append(plan.stops[-1])

    return added


def walked_to(stop: Stop, prev: Stop | None, predecessor: dict[str, str], rules: PlanningRules) -> Stop:
    """Keep a stop at its own times, its walk recomputed when the stop before changed.

    Args:
        stop: The kept stop.
        prev: The stop before it now, if any.
        predecessor: The id of the stop before each stored stop, by stop id.
        rules: The planning rules.

    Returns:
        The stop. With a new predecessor, the visitor arrives as the visit starts.
    """
    same_stop_before = predecessor.get(stop.poi_id) == (prev.poi_id if prev else None)
    if same_stop_before:
        return stop

    walk = rules.walking.minutes(prev.poi.location, stop.poi.location) if prev else 0

    return stop.model_copy(update={"walk_minutes": walk, "arrive": stop.start})


def emptied_per_gap(kept: Sequence[Stop], stored: Sequence[Stop]) -> list[int]:
    """Count the stored stops removed from each gap: before the first kept stop, between two, after the last."""
    kept_ids = {stop.poi_id for stop in kept}
    emptied = [0] * (len(kept) + 1)
    gap = 0

    for stop in stored:
        if stop.poi_id in kept_ids:
            gap += 1
        else:
            emptied[gap] += 1

    return emptied


def fill_gaps(
    kept: Sequence[Stop],
    stored: Sequence[Stop],
    window: TimeWindow,
    remaining: list[POI],
    rules: PlanningRules,
    *,
    wet: tuple[Interval, ...],
) -> list[Stop]:
    """Fill each gap around the kept stops from `remaining`, in order.

    A gap leaves room for the stops removed from the gaps after it, so an emptied slot is refilled before an earlier
    gap takes the whole budget.

    Args:
        kept: The stops that stay.
        stored: The stored stops, in order.
        window: The time window.
        remaining: The candidates left.
        rules: The planning rules.
        wet: The rain periods.

    Returns:
        The kept stops and the added ones, in order.
    """
    predecessor = {b.poi_id: a.poi_id for a, b in zip(stored, stored[1:])}
    emptied = emptied_per_gap(kept, stored)
    stops: list[Stop] = []
    fillers = 0

    for gap, nxt in enumerate([*kept, None]):
        prev = stops[-1] if stops else None
        builder = GapBuilder(window, rules, until=nxt, wet=wet)
        room = rules.max_stops - len(kept) - fillers - sum(emptied[gap + 1 :])
        filled = fill_gap(builder, gap_start(prev, window, rules), remaining, room)
        stops += filled
        fillers += len(filled)

        log.debug("gap before %s: %s", nxt.name if nxt else "the window's end", [stop.name for stop in filled])

        if nxt is not None:
            stops.append(walked_to(nxt, stops[-1] if stops else None, predecessor, rules))

    return stops


def refill(
    itinerary: Itinerary | None,
    window: TimeWindow,
    candidates: Sequence[POI],
    rules: PlanningRules,
    change: Change = NO_CHANGE,
    *,
    selected: set[str] = frozenset(),
) -> Rebuild:
    """Rebuild the plan for `window` after `change`.

    The stored stops that still fit keep their times, and every gap around them is filled from `candidates`. With no
    stored plan, a new plan.

    Args:
        itinerary: The stored plan, if any.
        window: The time window.
        candidates: The places to fill gaps with, best first.
        rules: The planning rules.
        change: What the message asks.
        selected: The ids the current preferences select, read when `change.recheck_preferences` is set.

    Returns:
        The new plan, with the stops removed and added.
    """
    stored = list(itinerary.stops) if itinerary else []
    kept, removed = split_stops(stored, window, selected, change)
    kept_ids = [stop.poi_id for stop in kept]
    unchanged = itinerary is not None and not removed and window == itinerary.window
    if unchanged:
        log.info("every stop still fits; the plan is unchanged")
        return Rebuild(itinerary=itinerary)

    taken = {stop.poi_id for stop in stored}
    remaining = [poi for poi in candidates if poi.id not in taken]
    stops = fill_gaps(kept, stored, window, remaining, rules, wet=change.wet)

    added = [stop.name for stop in stops if stop.poi_id not in kept_ids]
    rebuilt = Itinerary(window=window, stops=stops, assumptions=rules.assumptions())

    log.info("kept %s; removed %s; added %s", kept_ids, [(r.poi_id, r.reason) for r in removed], added)

    return Rebuild(itinerary=rebuilt, removed=removed, added=added)
