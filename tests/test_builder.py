import pytest

from tests.conftest import MON, TUE, WED, RULES, window
from corga.data.timeutil import at
from corga.planning.checks import validate
from corga.planning.models import Preferences
from corga.planning.replan import adapt_for_preferences, rebuild_from_first_misfit, replace_stop_for_preferences
from corga.planning.builder import replace_stop, build_itinerary, check_requested
from corga.planning.candidates import passes_filters, recommendations, select_candidates


@pytest.fixture
def history_pool(store):
    """The places a history interest selects, ranked."""
    return select_candidates(store.all(), Preferences(interests=["history"]))


@pytest.fixture
def history_walk(history_pool):
    """The Tuesday 11:00-15:00 history plan."""
    return build_itinerary(window(TUE, "11:00", "15:00"), history_pool, RULES)


def test_filters_child_museums_and_setting(store):
    prefs = Preferences(interests=["history"], with_child=True, avoid_museums=True, setting="outdoor")
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
    assert validate(it, RULES).ok


def test_historical_walk_fills_the_window(history_walk):
    assert history_walk.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-007"]
    assert history_walk.stops[-1].end <= history_walk.window.end
    assert validate(history_walk, RULES).ok


def test_builder_never_schedules_unknown_hours(store):
    it = build_itinerary(window(WED, "11:00", "15:00"), [store.get("CF-004"), store.get("CF-007")], RULES)
    assert it.poi_ids == ["CF-007"]


def test_replace_second_stop_for_a_child(store, history_walk):
    prefs = Preferences(interests=["history"], with_child=True)
    new = replace_stop(history_walk, 2, select_candidates(store.all(), prefs), RULES)
    assert new.poi_ids == ["CF-001", "CF-009", "CF-005", "CF-006", "CF-007"]
    assert store.get(new.poi_ids[1]).family_suitable is True
    assert new.stops[-1].end <= at(TUE, "15:00")


def test_replace_returns_none_when_nothing_fits(store, history_walk):
    assert replace_stop(history_walk, 2, [store.get("CF-021")], RULES) is None


def test_replace_rejects_bad_index(store, history_walk):
    with pytest.raises(ValueError):
        replace_stop(history_walk, 9, [], RULES)


def test_five_places_in_90_minutes_are_infeasible_with_reasons_and_alternative(store):
    pois = [store.get(i) for i in ["CF-001", "CF-002", "CF-021", "CF-022", "CF-017"]]
    report = check_requested(pois, window(TUE, "11:00", "12:30"), RULES)
    assert not report.feasible and report.visit_minutes == 355 and report.available_minutes == 90
    found = {(v.code, v.poi_id) for v in report.issues}
    assert ("TIME_BUDGET", None) in found
    assert ("CLOSED_DAY", "CF-002") in found
    assert ("LEG_TOO_LONG", "CF-021") in found
    assert report.alternative.poi_ids == ["CF-001"]


def test_feasible_request_returns_itinerary(store):
    pois = [store.get("CF-012"), store.get("CF-007")]
    report = check_requested(pois, window(WED, "13:00", "15:00"), RULES)
    assert report.feasible and sorted(report.itinerary.poi_ids) == ["CF-007", "CF-012"]


def test_outdoor_includes_mixed_places_and_indoor_does_not(store):
    fortress = store.get("CF-001")  # labelled mixed
    assert fortress.setting == "mixed"
    assert passes_filters(fortress, Preferences(setting="outdoor"))
    assert not passes_filters(fortress, Preferences(setting="indoor"))


def test_replan_keeps_the_stops_before_the_first_misfit(history_pool, history_walk):
    original = history_walk
    second = original.stops[1].poi_id
    candidates = [p for p in history_pool if p.id != second]
    new = rebuild_from_first_misfit(original, lambda s: s.poi_id != second, lambda: candidates, RULES)
    assert new.poi_ids[0] == original.poi_ids[0] and second not in new.poi_ids
    assert new.window == original.window


def test_preference_replan_replaces_stops_the_preferences_no_longer_select(store, history_walk):
    original = history_walk
    outdoor = select_candidates(store.all(), Preferences(interests=["history"], setting="outdoor"))
    replan = adapt_for_preferences(original, outdoor, RULES)
    assert replan.changed and replan.note.startswith(f"Replaced {len(replan.replaced)} stop(s)")
    first = original.poi_ids.index(replan.replaced[0])
    kept, added = replan.itinerary.poi_ids[:first], replan.itinerary.poi_ids[first:]
    assert kept == original.poi_ids[:first]  # the stops before the first misfit stay
    assert set(added) <= {p.id for p in outdoor}  # the rest come from what the preferences select
    assert replan.itinerary.window == original.window


def test_preference_replan_keeps_a_plan_that_still_matches(history_pool, history_walk):
    replan = adapt_for_preferences(history_walk, history_pool, RULES)
    assert not replan.changed and replan.itinerary == history_walk
    assert replan.note == "Every stop still matches the preferences."


def test_replan_changes_nothing_and_looks_up_nothing_when_every_stop_fits(history_walk):
    def no_lookup():
        raise AssertionError("candidates looked up although every stop fits")

    assert rebuild_from_first_misfit(history_walk, lambda s: True, no_lookup, RULES) == history_walk


def test_follow_up_replaces_stop_2_with_a_family_place(store, history_walk):
    """Multi-turn edit, planned directly: stop 2 changes, the rest and the 15:00 end are kept."""
    original = history_walk
    child = Preferences(interests=["history"], with_child=True)
    replan = replace_stop_for_preferences(original, 2, store.all(), child, rules=RULES)
    new = replan.itinerary
    assert new.poi_ids[1] != original.poi_ids[1] and store.get(new.poi_ids[1]).family_suitable is True
    assert [pid for n, pid in enumerate(new.poi_ids) if n != 1] == [p for n, p in enumerate(original.poi_ids) if n != 1]
    assert new.stops[-1].end <= at(TUE, "15:00") and validate(new, RULES).ok
    assert replan.changed and replan.replaced == [original.poi_ids[1]]
    assert replan.note == f"Replaced stop 2, {original.stops[1].name}, with {new.stops[1].name}."


def test_replacement_falls_back_to_ignoring_the_interests(store, history_walk):
    original = history_walk
    museums = Preferences(interests=["museum"])
    assert replace_stop(original, 2, select_candidates(store.all(), museums), RULES) is None  # no museum fits
    replan = replace_stop_for_preferences(original, 2, store.all(), museums, rules=RULES)
    assert replan.changed and replan.itinerary.poi_ids[1] != original.poi_ids[1]
    assert replan.note.endswith("No place matching the interests fits; it matches the other preferences.")


def test_a_stop_nothing_can_replace_leaves_the_plan_unchanged(store, history_walk):
    replan = replace_stop_for_preferences(history_walk, 2, [store.get("CF-021")], Preferences(), rules=RULES)
    assert not replan.changed and replan.replaced == [] and replan.itinerary == history_walk
    name = history_walk.stops[1].name
    assert replan.note == f"No alternative for stop 2, {name}, fits the window and constraints."


def test_recommendations_keep_only_places_open_on_the_day(store):
    museums = recommendations(store.all(), Preferences(interests=["museum"]), day=MON, limit=50)
    assert museums and not {"CF-014", "CF-017", "CF-031"} & {p.id for p in museums}  # closed on Mondays
    assert all(p.is_open_on(MON) for p in museums)
    assert len(recommendations(store.all(), Preferences(interests=["museum"]))) <= 5
