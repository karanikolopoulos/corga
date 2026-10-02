from typing import Self, NamedTuple
from datetime import date, time, datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Athens")


class Interval(NamedTuple):
    """A span of local time, [start, end)."""

    start: datetime
    end: datetime

    def overlaps(self, other: Self) -> bool:
        return self.start < other.end and other.start < self.end

    def meets(self, other: Self) -> bool:
        """Check `other` starts where this span ends."""
        return self.end == other.start

    def contains(self, other: Self) -> bool:
        return self.start <= other.start and other.end <= self.end


def at(day: date, hhmm: str | time) -> datetime:
    t = hhmm if isinstance(hhmm, time) else time.fromisoformat(hhmm)

    return datetime.combine(day, t, tzinfo=TZ)


def now() -> datetime:
    return datetime.now(TZ)


def hours_minutes(delta: timedelta) -> tuple[int, int]:
    return divmod(int(delta.total_seconds() // 60), 60)


def iso_duration(delta: timedelta) -> str:
    """Write a duration in ISO 8601: PT24H, PT1H25M, PT25M."""
    hours, minutes = hours_minutes(delta)
    if not hours:
        return f"PT{minutes}M"
    if not minutes:
        return f"PT{hours}H"

    return f"PT{hours}H{minutes}M"


def duration_text(delta: timedelta) -> str:
    """Write a duration for a reader: 24h, 1h 25m, 25m."""
    hours, minutes = hours_minutes(delta)
    if not hours:
        return f"{minutes}m"
    if not minutes:
        return f"{hours}h"

    return f"{hours}h {minutes}m"
