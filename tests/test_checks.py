from datetime import timedelta

import pytest

from tests.conftest import TUE, WED, RULES, window
from corga.data.timeutil import at
from corga.planning.checks import validate
from corga.planning.models import Itinerary
from corga.planning.schedule import Schedule


def plan(store, ids, day, start, end):
    scenario = window(day, start, end)
    stops = list(Schedule.of([store.get(i) for i in ids], scenario.start, RULES.walking))
    return Itinerary(window=scenario, stops=stops)


def codes(result):
    return {v.code for v in result.violations}


def test_window_must_be_ordered():
    with pytest.raises(ValueError):
        window(WED, "15:00", "13:00")


def test_schedule_walks_then_visits(store):
    it = plan(store, ["CF-007", "CF-012"], WED, "13:00", "17:00")
    first, second = it.stops
    assert (first.start, first.end) == (at(WED, "13:00"), at(WED, "13:55"))
    assert second.walk_minutes == 1 and second.arrive == at(WED, "13:56")


def test_valid_short_plan(store):
    result = validate(plan(store, ["CF-007", "CF-012"], WED, "13:00", "17:00"), store, RULES)
    assert result.ok, result.violations


def test_unknown_hours_are_never_promised(store):
    result = validate(plan(store, ["CF-004"], WED, "11:00", "15:00"), store, RULES)
    assert not result.ok and "UNKNOWN_HOURS" in codes(result)


def test_five_places_in_90_minutes_are_invalid_for_several_reasons(store):
    it = plan(store, ["CF-001", "CF-002", "CF-021", "CF-022", "CF-017"], TUE, "11:00", "12:30")
    result = validate(it, store, RULES)
    assert {"OUTSIDE_WINDOW", "CLOSED", "LEG_TOO_LONG"} <= codes(result)
    assert any(v.code == "CLOSED" and v.poi_id == "CF-002" for v in result.violations)


def test_travel_time_cannot_be_compressed(store):
    it = plan(store, ["CF-001", "CF-017"], WED, "11:00", "15:00")
    second = it.stops[1]
    it.stops[1] = second.model_copy(update={"arrive": second.arrive - timedelta(minutes=5)})
    assert "TRAVEL_TOO_SHORT" in codes(validate(it, store, RULES))


def test_empty_itinerary_is_an_error(store):
    assert "EMPTY" in codes(validate(Itinerary(window=window(WED, "13:00", "17:00")), store, RULES))
