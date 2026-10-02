from datetime import date, time

from corga.agent.turn import resolve_window, merge_preferences, unsupported_topics
from corga.agent.state import ParsedRequest
from corga.data.timeutil import at
from corga.planning.models import TimeWindow, Preferences

TUE = date(2026, 10, 6)


def test_full_window_from_request():
    req = ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(15))
    assert resolve_window(None, req) == TimeWindow(start=at(TUE, "11:00"), end=at(TUE, "15:00"))


def test_partial_request_updates_previous_window():
    prev = TimeWindow(start=at(TUE, "11:00"), end=at(TUE, "15:00"))
    assert resolve_window(prev, ParsedRequest(intent="plan", end_time=time(14))).end == at(TUE, "14:00")
    assert resolve_window(prev, ParsedRequest(intent="edit_plan")) == prev


def test_incomplete_first_window_is_none_and_inverted_keeps_previous():
    assert resolve_window(None, ParsedRequest(intent="plan", date=TUE)) is None
    prev = TimeWindow(start=at(TUE, "11:00"), end=at(TUE, "15:00"))
    assert resolve_window(prev, ParsedRequest(intent="plan", start_time=time(16))) == prev


def test_preferences_are_sticky():
    prefs = Preferences(interests=["history"])
    merged = merge_preferences(prefs, ParsedRequest(intent="edit_plan", stop_index=2, with_child=True))
    assert merged.interests == ["history"] and merged.with_child is True
    assert merge_preferences(merged, ParsedRequest(intent="plan", interests=["Art"])).interests == ["art"]


def test_unsupported_topics():
    assert unsupported_topics("Can I buy tickets online?") == ["tickets"]
    assert unsupported_topics("Is there a bookshop near the Liston?") == []
    assert unsupported_topics("Can I reserve a table?") == ["bookings"]
