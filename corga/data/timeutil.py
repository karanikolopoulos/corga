"""Europe/Athens time helpers; every datetime in CorGA is tz-aware local time."""

from datetime import date, time, datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Athens")


def at(day: date, hhmm: str | time) -> datetime:
    """Return a tz-aware Europe/Athens datetime for a local date and HH:MM."""
    t = hhmm if isinstance(hhmm, time) else time.fromisoformat(hhmm)
    return datetime.combine(day, t, tzinfo=TZ)


def now() -> datetime:
    return datetime.now(TZ)
