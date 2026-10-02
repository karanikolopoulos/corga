from datetime import time, timedelta

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from tests.conftest import TUE, WED, window, itinerary
from corga.agent.state import recent
from corga.agent.tools import ForecastRangeError, forecast_span
from corga.agent.request import PlanRequest, change_of, resolve_window, merge_preferences
from corga.domain.timeutil import Interval, at
from corga.domain.planning.models import Preferences
from corga.domain.planning.refill import Change


def test_partial_request_updates_previous_window():
    prev = window(TUE, "11:00", "15:00")

    assert resolve_window(prev, PlanRequest(end_time=time(14))).end == at(TUE, "14:00")
    assert resolve_window(prev, PlanRequest()) == prev


def test_preferences_are_sticky():
    prefs = Preferences(interests=["history"])
    merged = merge_preferences(prefs, PlanRequest(replace_stops=[2], child="yes"))

    assert merged.interests == ["history"] and merged.with_child is True
    assert merge_preferences(merged, PlanRequest(interests=["art"])).interests == ["art"]


def test_recent_keeps_the_last_five_messages_from_a_user_message():
    turns = [HumanMessage("h1"), AIMessage("a1"), HumanMessage("h2"), AIMessage("a2"), HumanMessage("h3")]

    assert [m.text for m in recent([*turns, AIMessage("a3"), HumanMessage("h4")])] == ["h2", "a2", "h3", "a3", "h4"]

    two_replies = [HumanMessage("h1"), AIMessage("a1"), AIMessage("a1b"), HumanMessage("h2"), AIMessage("a2")]

    assert [m.text for m in recent([*two_replies, HumanMessage("h3")])] == ["h2", "a2", "h3"]


def test_what_a_request_asks_of_the_stored_plan():
    assert change_of(PlanRequest(replace_stops=[2], child="yes")) == Change(replace=frozenset({2}))
    assert change_of(PlanRequest(avoid_museums="yes")) == Change(recheck_preferences=True)
    assert change_of(PlanRequest(keep_other_stops="no")).keep_others is False


def test_forecast_span_for_the_window_a_stop_a_day_and_the_next_hours(store):
    visit = window(WED, "13:00", "17:00")
    plan = itinerary(store, ["CF-001", "CF-003"], visit)
    now = at(WED, "09:20")

    assert forecast_span(visit, plan, now, stop=1) == plan.stops[0].span
    assert forecast_span(visit, None, now, visit_window=True) == visit.span  # no plan built yet
    assert forecast_span(None, None, now, day=WED) == Interval(at(WED, "00:00"), at(WED, "00:00") + timedelta(days=1))
    assert forecast_span(None, None, now, next_hours=3) == Interval(at(WED, "09:00"), at(WED, "12:00"))


def test_forecast_span_needs_a_stored_window_or_plan_and_a_stop_it_has(store):
    plan = itinerary(store, ["CF-001", "CF-003"], window(WED, "13:00", "17:00"))

    with pytest.raises(ForecastRangeError, match="no time window is stored"):
        forecast_span(None, None, at(WED, "09:00"), visit_window=True)

    with pytest.raises(ForecastRangeError, match="no plan is stored"):
        forecast_span(None, None, at(WED, "09:00"), stop=1)

    with pytest.raises(ForecastRangeError, match="stops 1-2"):
        forecast_span(plan.window, plan, at(WED, "09:00"), stop=3)

    with pytest.raises(ForecastRangeError, match="give one of"):
        forecast_span(plan.window, plan, at(WED, "09:00"))
