import pytest

from tests.conftest import WED, RULES, window
from corga.data.timeutil import at
from corga.weather.models import wet_periods
from corga.planning.checks import validate
from corga.planning.models import Preferences
from corga.planning.replan import is_exposed, adapt_for_weather
from corga.weather.fixture import FixtureWeather
from corga.planning.builder import build_itinerary
from corga.planning.candidates import select_candidates

WINDOW = window(WED, "13:00", "17:00")


@pytest.fixture
def outdoor_history(store):
    """The Wednesday 13:00-17:00 outdoor history plan."""
    prefs = Preferences(interests=["history"], setting="outdoor")
    return build_itinerary(WINDOW, select_candidates(store.all(), prefs), RULES)


@pytest.fixture
def rain_hours():
    """The fixture forecast for the rain window: dry until 15:00, rain after."""
    return FixtureWeather().hourly(WINDOW.start, WINDOW.end)


def test_fixture_rain_period_is_merged(rain_hours):
    periods = wet_periods(rain_hours)
    assert [(p.start, p.end) for p in periods] == [(at(WED, "15:00"), at(WED, "17:00"))]


def test_rain_replaces_exposed_stops_with_indoor_ones(store, outdoor_history, rain_hours):
    original = outdoor_history
    assert original.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-009"]
    periods = wet_periods(rain_hours)
    indoor = select_candidates(store.all(), Preferences(interests=["history"]).indoor())
    result = adapt_for_weather(original, periods, lambda: indoor, RULES)

    assert result.changed
    assert result.itinerary.poi_ids == ["CF-001", "CF-003", "CF-007"]
    assert set(result.replaced) == {"CF-005", "CF-006", "CF-009"}
    assert not any(is_exposed(s, periods) for s in result.itinerary.stops)
    assert result.itinerary.window == WINDOW
    assert validate(result.itinerary, RULES).ok


def test_dry_forecast_leaves_plan_unchanged(outdoor_history, rain_hours):
    original = outdoor_history
    dry = [h.model_copy(update={"precipitation_mm": 0.0, "weather_code": 1}) for h in rain_hours]

    def no_lookup():
        raise AssertionError("candidates looked up although no stop is exposed")

    result = adapt_for_weather(original, wet_periods(dry), no_lookup, RULES)
    assert not result.changed and result.itinerary == original
