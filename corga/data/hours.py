"""Fixture opening hours: null weekday = closed, end < start = overnight, 24:00 = end of day."""

from typing import Literal, NamedTuple
from datetime import date, datetime, timedelta

from pydantic import RootModel

from corga.data.timeutil import at

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
Access = Literal["open", "closed", "unknown"]


class Interval(NamedTuple):
    start: datetime
    end: datetime


def parse_range(day: date, spec: str) -> Interval:
    start_s, end_s = spec.split("-")
    start = at(day, start_s)
    end = at(day + timedelta(days=1), "00:00") if end_s == "24:00" else at(day, end_s)
    if end <= start:
        end = at(day + timedelta(days=1), end_s)
    return Interval(start, end)


class OpeningHours(RootModel[dict[str, str | None]]):
    """Weekday → "HH:MM-HH:MM" or None, exactly as in the dataset."""

    def __getitem__(self, weekday: str) -> str | None:
        return self.root[weekday]

    def intervals_starting_on(self, day: date) -> list[Interval]:
        spec = self.root.get(WEEKDAYS[day.weekday()])
        return [parse_range(day, spec)] if spec else []

    def intervals_around(self, day: date) -> list[Interval]:
        """Intervals of the previous day (overnight spill) and of the day itself."""
        return self.intervals_starting_on(day - timedelta(days=1)) + self.intervals_starting_on(day)

    def is_open_on(self, day: date) -> bool:
        return bool(self.intervals_starting_on(day))

    def covers(self, start: datetime, end: datetime) -> bool:
        return any(i.start <= start and end <= i.end for i in self.intervals_around(start.date()))

    def earliest_start(self, arrive: datetime, minutes: int) -> datetime | None:
        """Earliest visit start >= arrive such that the whole visit fits inside one interval."""
        need = timedelta(minutes=minutes)
        for interval in sorted(self.intervals_around(arrive.date())):
            start = max(arrive, interval.start)
            if start + need <= interval.end:
                return start
        return None


def access(hours: OpeningHours | None, start: datetime, end: datetime) -> Access:
    if hours is None:
        return "unknown"
    return "open" if hours.covers(start, end) else "closed"
