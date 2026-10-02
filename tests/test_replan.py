from datetime import date

from corga.paths import RAIN_FIXTURE_PATH
from corga.data.timeutil import at
from corga.planning.rules import PlanningRules
from corga.weather.replan import is_exposed, wet_periods, adapt_for_weather
from corga.planning.checks import validate
from corga.planning.models import TimeWindow, Preferences
from corga.weather.fixture import FixtureWeather
from corga.planning.builder import build_itinerary
from corga.planning.candidates import select_candidates

WED = date(2026, 10, 7)
RULES = PlanningRules()
WINDOW = TimeWindow(start=at(WED, "13:00"), end=at(WED, "17:00"))


def outdoor_history(store):
    prefs = Preferences(interests=["history"], outdoor_only=True)
    return build_itinerary(WINDOW, select_candidates(store.all(), prefs), RULES)


def test_fixture_rain_period_is_merged():
    periods = wet_periods(FixtureWeather(RAIN_FIXTURE_PATH).hourly(WINDOW.start, WINDOW.end))
    assert [(p.start, p.end) for p in periods] == [(at(WED, "15:00"), at(WED, "17:00"))]


def test_rain_replaces_exposed_stops_with_indoor_ones(store):
    original = outdoor_history(store)
    assert original.poi_ids == ["CF-002", "CF-003", "CF-005", "CF-006", "CF-009"]
    hours = FixtureWeather(RAIN_FIXTURE_PATH).hourly(WINDOW.start, WINDOW.end)
    indoor = select_candidates(store.all(), Preferences(interests=["history"], indoor_only=True))
    result = adapt_for_weather(original, hours, store, indoor, RULES)

    assert result.changed
    assert result.itinerary.poi_ids == ["CF-002", "CF-003", "CF-007", "CF-017"]
    assert set(result.replaced) == {"CF-005", "CF-006", "CF-009"}
    assert not any(is_exposed(s, result.wet_periods) for s in result.itinerary.stops)
    assert result.itinerary.window == WINDOW
    assert validate(result.itinerary, store, RULES).ok


def test_dry_forecast_leaves_plan_unchanged(store):
    original = outdoor_history(store)
    hours = FixtureWeather(RAIN_FIXTURE_PATH).hourly(WINDOW.start, WINDOW.end)
    dry = [h.model_copy(update={"precipitation_mm": 0.0, "weather_code": 1}) for h in hours]
    result = adapt_for_weather(original, dry, store, [], RULES)
    assert not result.changed and result.itinerary == original
