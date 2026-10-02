"""The deterministic glue between a parsed request and the stored state: the time window, the preferences, the hours
to forecast, and the topics the dataset cannot answer."""

import re

from datetime import datetime, timedelta

from corga.agent.state import AgentState, ParsedRequest
from corga.data.timeutil import at
from corga.planning.models import TimeWindow, Preferences

UNSUPPORTED = {
    "tickets": r"\b(tickets?|admission|entry fees?|prices?|costs?)\b",
    "bookings": r"\b(book|booking|reserve|reservations?)\b",
    "accessibility": r"\b(wheelchair|accessible|accessibility)\b",
    "events": r"\b(events?|concerts?|festivals?)\b",
}


def resolve_window(prev: TimeWindow | None, req: ParsedRequest) -> TimeWindow | None:
    """The time window after this message: the previous one with any stated date or times replaced."""
    if not (req.date or req.start_time or req.end_time):
        return prev  # nothing stated, keep the window
    if prev is None and not (req.date and req.start_time and req.end_time):
        return None  # a first window needs all three parts
    day = req.date or prev.day
    start = at(day, req.start_time or prev.start.time())
    end = at(day, req.end_time or prev.end.time())
    return TimeWindow(start=start, end=end) if end > start else prev  # an inverted edit is ignored


def merge_preferences(prefs: Preferences, req: ParsedRequest) -> Preferences:
    """Preferences stated in this message override the stored ones; unstated ones are kept."""
    return prefs.merged(req.preference_updates)


def weather_range(state: AgentState, now: datetime) -> tuple[datetime, datetime]:
    """The hours to forecast: a stated horizon, else the plan's window, else the stated day, else the next 24 h."""
    request = state.request
    this_hour = now.replace(minute=0, second=0, microsecond=0)
    if request.horizon_hours:
        return this_hour, this_hour + timedelta(hours=request.horizon_hours)
    windows: tuple[TimeWindow | None, ...] = (state.itinerary.window if state.itinerary else None, state.window)
    for window in windows:
        if window is not None and (request.date is None or request.date == window.day):
            return window.start, window.end
    if request.date is not None:
        return at(request.date, request.start_time or "00:00"), at(request.date, request.end_time or "23:59")
    return this_hour, this_hour + timedelta(hours=24)


def unsupported_topics(text: str) -> list[str]:
    """Topics in the text that the dataset has no facts about: tickets, bookings, accessibility, events."""
    return [topic for topic, pattern in UNSUPPORTED.items() if re.search(pattern, text, re.IGNORECASE)]
