"""Europe/Athens time helpers; every datetime in CorGA is tz-aware local time."""

from typing import NamedTuple
from datetime import date, time, datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Athens")


class Interval(NamedTuple):
    """A span of local time, [start, end)."""

    start: datetime
    end: datetime

    def overlaps(self, other: "Interval") -> bool:
        """The two spans share some time."""
        return self.start < other.end and other.start < self.end

    def contains(self, other: "Interval") -> bool:
        """`other` lies entirely inside this span."""
        return self.start <= other.start and other.end <= self.end


def at(day: date, hhmm: str | time) -> datetime:
    """Return a tz-aware Europe/Athens datetime for a local date and HH:MM."""
    t = hhmm if isinstance(hhmm, time) else time.fromisoformat(hhmm)
    return datetime.combine(day, t, tzinfo=TZ)


def now() -> datetime:
    return datetime.now(TZ)
