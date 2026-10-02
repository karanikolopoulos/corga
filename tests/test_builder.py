from datetime import date

import pytest

from corga.data.timeutil import at
from corga.planning.rules import PlanningRules
from corga.planning.checks import validate
from corga.planning.models import TimeWindow, Preferences
from corga.planning.builder import replace_stop, build_itinerary, check_requested
from corga.planning.candidates import select_candidates

MON, TUE, WED = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)
RULES = PlanningRules()


def window(day, start, end):
    return TimeWindow(start=at(day, start), end=at(day, end))


def test_filters_child_museums_and_setting(store):
    prefs = Preferences(interests=["history"], with_child=True, avoid_museums=True, outdoor_only=True)
    picked = select_candidates(store.all(), prefs)
    assert picked and all(
        p.family_suitable is True and "museum" not in p.tags and p.setting == "outdoor" and p.category == "history"
        for p in picked
    )


def test_art_and_museums_use_tags_not_category(store):
    ids = {p.id for p in select_candidates(store.all(), Preferences(interests=["art", "museums"]))}
    assert {"CF-007", "CF-014", "CF-017", "CF-045"} <= ids


def test_ranker_reorders_candidates(store):
    prefs = Preferences(interests=["venetian"])
    picked = select_candidates(store.all(), prefs, ranker=lambda text, allowed: ["CF-028", "CF-001"])
    assert [p.id for p in picked[:2]] == ["CF-028", "CF-001"]


def test_builder_skips_monday_closures(store):
    candidates = [store.get(i) for i in ["CF-014", "CF-017", "CF-007", "CF-012", "CF-029"]]
    it = build_itinerary(window(MON, "13:00", "17:00"), candidates, RULES)
    assert it.poi_ids == ["CF-007", "CF-012", "CF-029"]
    assert validate(it, store, RULES).ok


def test_historical_walk_fills_the_window(store):
    w = window(TUE, "11:00", "15:00")
    it = build_itinerary(w, select_candidates(store.all(), Preferences(interests=["history"])), RULES)
    assert it.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-007"]
    assert it.stops[-1].end <= w.end
    assert validate(it, store, RULES).ok


def test_builder_never_schedules_unknown_hours(store):
    it = build_itinerary(window(WED, "11:00", "15:00"), [store.get("CF-004"), store.get("CF-007")], RULES)
    assert it.poi_ids == ["CF-007"]


@pytest.fixture
def history_walk(store):
    w = TimeWindow(start=at(TUE, "11:00"), end=at(TUE, "15:00"))
    return build_itinerary(w, select_candidates(store.all(), Preferences(interests=["history"])), RULES)


def test_replace_second_stop_for_a_child(store, history_walk):
    prefs = Preferences(interests=["history"], with_child=True)
    new = replace_stop(history_walk, 2, select_candidates(store.all(), prefs), store, RULES)
    assert new.poi_ids == ["CF-001", "CF-009", "CF-005", "CF-006", "CF-007"]
    assert store.get(new.poi_ids[1]).family_suitable is True
    assert new.stops[-1].end <= at(TUE, "15:00")


def test_replace_returns_none_when_nothing_fits(store, history_walk):
    assert replace_stop(history_walk, 2, [store.get("CF-021")], store, RULES) is None


def test_replace_rejects_bad_index(store, history_walk):
    with pytest.raises(ValueError):
        replace_stop(history_walk, 9, [], store, RULES)


def test_five_places_in_90_minutes_are_infeasible_with_reasons_and_alternative(store):
    pois = [store.get(i) for i in ["CF-001", "CF-002", "CF-021", "CF-022", "CF-017"]]
    report = check_requested(pois, TimeWindow(start=at(TUE, "11:00"), end=at(TUE, "12:30")), store, RULES)
    assert not report.feasible and report.visit_minutes == 355 and report.available_minutes == 90
    found = {(v.code, v.poi_id) for v in report.issues}
    assert ("TIME_BUDGET", None) in found
    assert ("CLOSED_DAY", "CF-002") in found
    assert ("LEG_TOO_LONG", "CF-021") in found
    assert report.alternative.poi_ids == ["CF-001"]


def test_feasible_request_returns_itinerary(store):
    pois = [store.get("CF-012"), store.get("CF-007")]
    window = TimeWindow(start=at(WED, "13:00"), end=at(WED, "15:00"))
    report = check_requested(pois, window, store, RULES)
    assert report.feasible and sorted(report.itinerary.poi_ids) == ["CF-007", "CF-012"]
