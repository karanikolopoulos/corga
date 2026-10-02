from datetime import date, datetime, timedelta

from pydantic import RootModel

from corga.domain.timeutil import Interval, at

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def parse_range(day: date, spec: str) -> Interval:
    """Read an "HH:MM-HH:MM" spec as an interval.

    Args:
        day: The day the interval starts.
        spec: The opening hours, as in the dataset.

    Returns:
        The interval. An end before the start runs into the next day.
    """
    start_s, end_s = spec.split("-")
    start = at(day, start_s)
    end = at(day + timedelta(days=1), "00:00") if end_s == "24:00" else at(day, end_s)
    if end <= start:
        end = at(day + timedelta(days=1), end_s)

    return Interval(start, end)


class OpeningHours(RootModel[dict[str, str | None]]):
    """Opening hours by weekday, "HH:MM-HH:MM" or None, exactly as in the dataset."""

    def __getitem__(self, weekday: str) -> str | None:
        return self.root[weekday]

    def __contains__(self, span: Interval) -> bool:
        """Check the whole span falls inside one opening interval."""
        return any(interval.contains(span) for interval in self.intervals_around(span.start.date()))

    def intervals_starting_on(self, day: date) -> list[Interval]:
        spec = self.root.get(WEEKDAYS[day.weekday()])

        return [parse_range(day, spec)] if spec else []

    def intervals_around(self, day: date) -> list[Interval]:
        """List the intervals of the day before (overnight spill) and of the day."""
        return self.intervals_starting_on(day - timedelta(days=1)) + self.intervals_starting_on(day)

    def is_open_on(self, day: date) -> bool:
        return bool(self.intervals_starting_on(day))

    def earliest_start(self, arrive: datetime, minutes: int) -> datetime | None:
        """Find the earliest visit start that fits one interval.

        Args:
            arrive: When the visitor arrives.
            minutes: How long the visit lasts.

        Returns:
            The start, at or after `arrive`, or None if no interval fits.
        """
        need = timedelta(minutes=minutes)

        for interval in sorted(self.intervals_around(arrive.date())):
            start = max(arrive, interval.start)
            if start + need <= interval.end:
                return start

        return None
