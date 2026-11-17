import pytest

from tests.conftest import TUE, WED, RULES, window
from corga.domain.weather import wet_periods
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.checks import validate
from corga.domain.planning.models import Preferences
from corga.domain.planning.refill import Change, refill, is_exposed
from corga.adapters.weather.fixture import FixtureWeather
from corga.domain.planning.candidates import select_candidates

HISTORY_WINDOW = window(TUE, "11:00", "15:00")
RAIN_WINDOW = window(WED, "13:00", "17:00")
HISTORY = Preferences(interests=["history"])


def pool(store, prefs):
    return select_candidates(store.all(), prefs)


@pytest.fixture
def history_walk(store):
    """Build the Tuesday 11:00-15:00 history walk."""
    return refill(None, HISTORY_WINDOW, pool(store, HISTORY), RULES).itinerary


def times(itinerary):
    return {stop.poi_id: (stop.start, stop.end) for stop in itinerary.stops}


def test_replacing_stop_2_keeps_the_other_stops_at_their_times(store, history_walk):
    child = HISTORY.merged({"with_child": True})
    rebuild = refill(history_walk, HISTORY_WINDOW, pool(store, child), RULES, Change(replace=frozenset({2})))
    new = rebuild.itinerary

    assert new.poi_ids == ["CF-001", "CF-009", "CF-005", "CF-006", "CF-007"]

    kept = ["CF-001", "CF-005", "CF-006", "CF-007"]
    assert {pid: times(new)[pid] for pid in kept} == {pid: times(history_walk)[pid] for pid in kept}
    assert (new.stops[2].walk_minutes, new.stops[2].arrive) == (5, new.stops[2].start)  # walked from the new stop 2
    assert [(r.poi_id, r.reason) for r in rebuild.removed] == [("CF-003", "replaced")]
    assert rebuild.added == ["Old Town Hall (Loggia dei Nobili)"] and store.get("CF-009").family_suitable is True
    assert validate(new, RULES).ok


def test_rechecked_preferences_remove_the_stops_they_no_longer_select(store, history_walk):
    prefs = HISTORY.merged({"with_child": True, "avoid_museums": True})
    selected = {poi.id for poi in pool(store, prefs)}
    recheck = Change(recheck_preferences=True)
    rebuild = refill(history_walk, HISTORY_WINDOW, pool(store, prefs), RULES, recheck, selected=selected)

    assert [(r.poi_id, r.reason) for r in rebuild.removed] == [("CF-001", "preferences"), ("CF-007", "preferences")]
    assert {"CF-003", "CF-005", "CF-006"} <= set(rebuild.itinerary.poi_ids) <= selected
    assert validate(rebuild.itinerary, RULES).ok


def test_rain_removes_the_exposed_stops_and_fills_the_time_indoors(store):
    outdoor = Preferences(interests=["history"], setting="outdoor")
    outdoor_plan = refill(None, RAIN_WINDOW, pool(store, outdoor), RULES).itinerary

    assert outdoor_plan.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-009"]

    wet = tuple(wet_periods(FixtureWeather().hourly(RAIN_WINDOW.start, RAIN_WINDOW.end)))
    rebuild = refill(outdoor_plan, RAIN_WINDOW, pool(store, outdoor.indoor()), RULES, Change(wet=wet))

    assert rebuild.itinerary.poi_ids == ["CF-001", "CF-003", "CF-007"]
    assert {r.poi_id: r.reason for r in rebuild.removed} == {"CF-005": "rain", "CF-006": "rain", "CF-009": "rain"}
    assert not any(is_exposed(stop, wet) for stop in rebuild.itinerary.stops)
    assert validate(rebuild.itinerary, RULES).ok


def test_a_shorter_window_removes_the_stops_outside_it(store, history_walk):
    rebuild = refill(history_walk, window(TUE, "11:00", "14:00"), pool(store, HISTORY), RULES)

    assert rebuild.itinerary.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006"]
    assert [(r.poi_id, r.reason) for r in rebuild.removed] == [("CF-007", "window")]
    assert rebuild.added == []


def test_a_longer_window_fills_the_new_time(store):
    three = refill(None, HISTORY_WINDOW, pool(store, HISTORY), PlanningRules(max_stops=3)).itinerary

    assert three.poi_ids == ["CF-001", "CF-003", "CF-005"]

    rebuild = refill(three, window(TUE, "11:00", "16:00"), pool(store, HISTORY), RULES)

    assert rebuild.itinerary.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-007"]
    assert rebuild.added == ["Maitland Rotunda", "Casa Parlante Museum"] and not rebuild.removed


def test_a_filler_must_reach_the_next_kept_stop_in_time(store, history_walk):
    far = [store.get("CF-021")]  # the Achilleion, outside Corfu Town
    rebuild = refill(history_walk, HISTORY_WINDOW, far, RULES, Change(replace=frozenset({2})))

    assert rebuild.itinerary.poi_ids == ["CF-001", "CF-005", "CF-006", "CF-007"]
    assert rebuild.added == [] and validate(rebuild.itinerary, RULES).ok


def test_each_emptied_slot_is_refilled_before_an_earlier_gap_takes_the_budget(store):
    six_hours = window(WED, "11:00", "17:00")
    plan = refill(None, six_hours, pool(store, Preferences()), RULES).itinerary
    child = Preferences(with_child=True)
    selected = {poi.id for poi in pool(store, child)}
    rebuild = refill(plan, six_hours, pool(store, child), RULES, Change(recheck_preferences=True), selected=selected)

    assert plan.poi_ids == ["CF-001", "CF-003", "CF-002", "CF-005", "CF-006"]
    assert rebuild.itinerary.poi_ids == ["CF-007", "CF-003", "CF-009", "CF-005", "CF-006"]  # one per fortress slot
    assert validate(rebuild.itinerary, RULES).ok
