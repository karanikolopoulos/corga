import pytest

from tests.conftest import MON, TUE, WED, RULES, window
from corga.data.timeutil import at
from corga.planning.checks import validate
from corga.planning.models import Preferences
from corga.planning.builder import replace_stop, build_itinerary, check_requested
from corga.planning.candidates import select_candidates


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
    w = window(TUE, "11:00", "15:00")
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
    report = check_requested(pois, window(TUE, "11:00", "12:30"), store, RULES)
    assert not report.feasible and report.visit_minutes == 355 and report.available_minutes == 90
    found = {(v.code, v.poi_id) for v in report.issues}
    assert ("TIME_BUDGET", None) in found
    assert ("CLOSED_DAY", "CF-002") in found
    assert ("LEG_TOO_LONG", "CF-021") in found
    assert report.alternative.poi_ids == ["CF-001"]


def test_feasible_request_returns_itinerary(store):
    pois = [store.get("CF-012"), store.get("CF-007")]
    report = check_requested(pois, window(WED, "13:00", "15:00"), store, RULES)
    assert report.feasible and sorted(report.itinerary.poi_ids) == ["CF-007", "CF-012"]


def test_outdoor_includes_mixed_places_and_indoor_does_not(store):
    from corga.planning.candidates import passes_filters

    fortress = store.get("CF-001")  # labelled mixed
    assert fortress.setting == "mixed"
    assert passes_filters(fortress, Preferences(setting="outdoor"))
    assert not passes_filters(fortress, Preferences(setting="indoor"))


def test_replan_keeps_the_stops_before_the_first_misfit(store):
    from corga.planning.builder import replan_from_first_misfit

    history = select_candidates(store.all(), Preferences(interests=["history"]))
    original = build_itinerary(window(TUE, "11:00", "15:00"), history, RULES)
    second = original.stops[1].poi_id
    candidates = [p for p in history if p.id != second]
    replan = replan_from_first_misfit(original, lambda s: s.poi_id != second, lambda: candidates, store, RULES)
    assert replan.changed and second in replan.replaced
    assert replan.itinerary.poi_ids[0] == original.poi_ids[0] and second not in replan.itinerary.poi_ids
    assert replan.itinerary.window == original.window


def test_replan_changes_nothing_and_looks_up_nothing_when_every_stop_fits(store):
    from corga.planning.builder import replan_from_first_misfit

    history = select_candidates(store.all(), Preferences(interests=["history"]))
    original = build_itinerary(window(TUE, "11:00", "15:00"), history, RULES)

    def no_lookup():
        raise AssertionError("candidates looked up although every stop fits")

    replan = replan_from_first_misfit(original, lambda s: True, no_lookup, store, RULES)
    assert not replan.changed and replan.itinerary == original


def test_follow_up_replaces_stop_2_with_a_family_place(store):
    """Multi-turn edit, planned directly: stop 2 changes, the rest and the 15:00 end are kept."""
    from corga.planning.builder import replace_stop_for_preferences

    history = Preferences(interests=["history"])
    original = build_itinerary(window(TUE, "11:00", "15:00"), select_candidates(store.all(), history), RULES)
    child = history.model_copy(update={"with_child": True})
    new = replace_stop_for_preferences(original, 2, store.all(), child, store=store, rules=RULES)
    assert new.poi_ids[1] != original.poi_ids[1] and store.get(new.poi_ids[1]).family_suitable is True
    assert [pid for n, pid in enumerate(new.poi_ids) if n != 1] == [p for n, p in enumerate(original.poi_ids) if n != 1]
    assert new.stops[-1].end <= at(TUE, "15:00") and validate(new, store, RULES).ok


def test_replacement_falls_back_to_ignoring_the_interests(store):
    from corga.planning.builder import replace_stop, replace_stop_for_preferences

    original = build_itinerary(
        window(TUE, "11:00", "15:00"), select_candidates(store.all(), Preferences(interests=["history"])), RULES
    )
    museums = Preferences(interests=["museum"])
    assert replace_stop(original, 2, select_candidates(store.all(), museums), store, RULES) is None  # no museum fits
    new = replace_stop_for_preferences(original, 2, store.all(), museums, store=store, rules=RULES)
    assert new is not None and new.poi_ids[1] != original.poi_ids[1]


def test_recommendations_keep_only_places_open_on_the_day(store):
    from corga.planning.candidates import recommendations

    museums = recommendations(store.all(), Preferences(interests=["museum"]), day=MON, limit=50)
    assert museums and not {"CF-014", "CF-017", "CF-031"} & {p.id for p in museums}  # closed on Mondays
    assert all(p.is_open_on(MON) for p in museums)
    assert len(recommendations(store.all(), Preferences(interests=["museum"]))) <= 5
