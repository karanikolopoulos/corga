from typing import Self
from datetime import datetime, timedelta
from dataclasses import replace, dataclass
from collections.abc import Iterator, Sequence

from corga.domain.places import POI
from corga.domain.planning.geo import WalkingModel
from corga.domain.planning.models import Stop


def visit_start(poi: POI, arrive: datetime) -> datetime:
    """Start the visit on arrival, or at opening time if it still fits.

    Args:
        poi: The place.
        arrive: When the visitor arrives.

    Returns:
        The start. The guards flag a visit that does not fit.
    """
    hours = poi.opening_hours
    if hours is None:
        return arrive

    return hours.earliest_start(arrive, poi.suggested_visit_minutes) or arrive


@dataclass(frozen=True)
class Schedule:
    """Stops visited in order, and when the visitor is free to walk on."""

    walking: WalkingModel
    clock: datetime  # when the visitor is free to walk on
    stops: tuple[Stop, ...] = ()

    @property
    def here(self) -> POI | None:
        """The last POI visited."""
        return self.stops[-1].poi if self.stops else None

    @classmethod
    def of(cls, pois: Sequence[POI], start_at: datetime, walking: WalkingModel) -> Self:
        """Schedule the places in the given order.

        Args:
            pois: The places.
            start_at: When the visitor sets off.
            walking: The walking model.

        Returns:
            The schedule.
        """
        schedule = cls(walking=walking, clock=start_at)

        for poi in pois:
            schedule = schedule.then(poi)

        return schedule

    def then(self, poi: POI) -> Self:
        """Add a place as the next stop, leaving this schedule unchanged.

        Args:
            poi: The place.

        Returns:
            The new schedule.
        """
        walk = self.walking.minutes(self.here.location, poi.location) if self.here else 0  # no walk to the first stop
        arrive = self.clock + timedelta(minutes=walk)  # set off when free, arrive after the walk
        start = visit_start(poi, arrive)  # wait for opening if needed
        end = start + timedelta(minutes=poi.suggested_visit_minutes)  # stay for the suggested visit
        stop = Stop(poi=poi, walk_minutes=walk, arrive=arrive, start=start, end=end)

        return replace(self, clock=end, stops=(*self.stops, stop))  # free again at `end`, now at `poi`

    def __iter__(self) -> Iterator[Stop]:
        yield from self.stops
