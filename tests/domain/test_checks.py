from datetime import timedelta

import pytest

from tests.conftest import TUE, WED, RULES, window, itinerary
from corga.domain.timeutil import at
from corga.domain.planning.checks import validate
from corga.domain.planning.models import Itinerary


def codes(result):
    return {v.code for v in result.violations}


def test_window_must_be_ordered():
    with pytest.raises(ValueError):
        window(WED, "15:00", "13:00")


def test_schedule_walks_then_visits(store):
    it = itinerary(store, ["CF-007", "CF-012"], window(WED, "13:00", "17:00"))
    first, second = it.stops
    assert (first.start, first.end) == (at(WED, "13:00"), at(WED, "13:55"))
    assert second.walk_minutes == 1 and second.arrive == at(WED, "13:56")


def test_valid_short_plan(store):
    result = validate(itinerary(store, ["CF-007", "CF-012"], window(WED, "13:00", "17:00")), RULES)
    assert result.ok, result.violations


def test_unknown_hours_are_never_promised(store):
    result = validate(itinerary(store, ["CF-004"], window(WED, "11:00", "15:00")), RULES)
    assert not result.ok and "UNKNOWN_HOURS" in codes(result)


def test_five_places_in_90_minutes_are_invalid_for_several_reasons(store):
    it = itinerary(store, ["CF-001", "CF-002", "CF-021", "CF-022", "CF-017"], window(TUE, "11:00", "12:30"))
    result = validate(it, RULES)
    assert {"OUTSIDE_WINDOW", "CLOSED", "LEG_TOO_LONG"} <= codes(result)
    assert any(v.code == "CLOSED" and v.poi_id == "CF-002" for v in result.violations)


def test_travel_time_cannot_be_compressed(store):
    it = itinerary(store, ["CF-001", "CF-017"], window(WED, "11:00", "15:00"))
    second = it.stops[1]
    it.stops[1] = second.model_copy(update={"arrive": second.arrive - timedelta(minutes=5)})
    assert "TRAVEL_TOO_SHORT" in codes(validate(it, RULES))


def test_violations_follow_the_visits_in_order(store):
    it = itinerary(store, ["CF-001", "CF-017", "CF-012"], window(WED, "11:00", "15:00"))
    second, third = it.stops[1], it.stops[2]
    it.stops[1] = second.model_copy(update={"arrive": second.arrive - timedelta(minutes=5)})  # the walk into stop 2
    it.stops[2] = third.model_copy(update={"end": third.start + timedelta(minutes=5)})  # stop 3 itself
    found = [(v.stop, v.code) for v in validate(it, RULES).violations]
    assert found == [(2, "TRAVEL_TOO_SHORT"), (3, "VISIT_TOO_SHORT")]


def test_empty_itinerary_is_an_error(store):
    assert "EMPTY" in codes(validate(Itinerary(window=window(WED, "13:00", "17:00")), RULES))


def test_a_place_visited_twice_is_a_duplicate_named_once(store):
    result = validate(itinerary(store, ["CF-007", "CF-012", "CF-007"], window(WED, "13:00", "17:00")), RULES)
    [duplicate] = [v for v in result.violations if v.code == "DUPLICATE"]
    assert not result.ok and duplicate.message == "Repeated stops: Casa Parlante Museum."
