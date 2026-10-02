from tests.conftest import MON, TUE, WED, RULES, window
from corga.domain.planning.checks import validate
from corga.domain.planning.models import Preferences
from corga.domain.planning.builder import build_itinerary, check_requested
from corga.domain.planning.candidates import passes_filters, select_candidates


def test_filters_child_museums_and_setting(store):
    prefs = Preferences(interests=["history"], with_child=True, avoid_museums=True, setting="outdoor")
    picked = select_candidates(store.all(), prefs)

    assert picked and all(
        p.family_suitable is True and "museum" not in p.tags and p.setting == "outdoor" and p.category == "history"
        for p in picked
    )


def test_builder_skips_monday_closures(store):
    candidates = [store.get(i) for i in ["CF-014", "CF-017", "CF-007", "CF-012", "CF-029"]]
    it = build_itinerary(window(MON, "13:00", "17:00"), candidates, RULES)

    assert it.poi_ids == ["CF-007", "CF-012", "CF-029"]
    assert validate(it, RULES).ok


def test_builder_never_schedules_unknown_hours(store):
    it = build_itinerary(window(WED, "11:00", "15:00"), [store.get("CF-004"), store.get("CF-007")], RULES)

    assert it.poi_ids == ["CF-007"]


def test_five_places_in_90_minutes_are_infeasible_with_reasons_and_alternative(store):
    pois = [store.get(i) for i in ["CF-001", "CF-002", "CF-021", "CF-022", "CF-017"]]
    report = check_requested(pois, window(TUE, "11:00", "12:30"), RULES)

    assert not report.feasible and report.visit_minutes == 355 and report.available_minutes == 90

    found = {(v.code, v.poi_id) for v in report.issues}
    assert ("TIME_BUDGET", None) in found
    assert ("CLOSED_DAY", "CF-002") in found
    assert ("LEG_TOO_LONG", "CF-021") in found
    assert report.alternative.poi_ids == ["CF-001"]


def test_outdoor_includes_mixed_places_and_indoor_does_not(store):
    fortress = store.get("CF-001")  # labelled mixed

    assert fortress.setting == "mixed"
    assert passes_filters(fortress, Preferences(setting="outdoor"))
    assert not passes_filters(fortress, Preferences(setting="indoor"))
