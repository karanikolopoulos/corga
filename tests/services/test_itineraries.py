"""The itinerary use cases on the in-memory dataset store and a plain search fake: no LangChain, no graph."""

import pytest

from tests.conftest import TUE, WED, window
from corga.services.results import RefusalResult, NeedInfoResult
from corga.services.itineraries import Itineraries
from corga.domain.planning.models import Preferences
from corga.adapters.weather.fixture import FixtureWeather

NAMED = ["Old Fortress", "New Fortress", "Achilleion Palace", "Mon Repos Museum of Palaiopolis", "Museum of Asian Art"]


@pytest.fixture
def itineraries(steps, places):
    return Itineraries(steps, places)


def test_plan_itinerary_asks_for_a_window(itineraries):
    assert itineraries.plan_itinerary(None, Preferences()) == NeedInfoResult(missing="a date plus start and end time")


def test_plan_itinerary_returns_a_validated_plan(itineraries):
    result = itineraries.plan_itinerary(window(TUE, "11:00", "15:00"), Preferences(interests=["history"]))
    assert result.itinerary.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-007"]
    assert result.validation.ok


def test_check_places_refuses_places_not_in_the_dataset(itineraries):
    result = itineraries.check_places(window(WED, "13:00", "16:00"), ["Old Fortress", "Corfu Aquarium"])
    assert result == RefusalResult(reason="not_in_dataset", names=["Corfu Aquarium"])


def test_check_places_explains_an_infeasible_request_and_offers_a_validated_alternative(itineraries):
    result = itineraries.check_places(window(TUE, "11:00", "12:30"), NAMED)
    assert result.feasibility.feasible is False and result.itinerary.poi_ids == ["CF-001"]
    assert result.validation.ok


@pytest.fixture
def history_walk(itineraries):
    """The Tuesday 11:00-15:00 history plan."""
    return itineraries.plan_itinerary(window(TUE, "11:00", "15:00"), Preferences(interests=["history"])).itinerary


def test_change_stop_asks_for_an_itinerary_and_a_stop_number(itineraries, history_walk):
    prefs = Preferences()
    assert itineraries.change_stop(None, 2, prefs) == NeedInfoResult(missing="an existing itinerary to edit")
    assert itineraries.change_stop(history_walk, None, prefs) == NeedInfoResult(missing="which stop to change (1-5)")
    assert itineraries.change_stop(history_walk, 6, prefs).kind == "need_info"


def test_change_stop_replaces_stop_2_for_a_child_and_keeps_the_rest(itineraries, history_walk):
    result = itineraries.change_stop(history_walk, 2, Preferences(interests=["history"], with_child=True))
    assert result.itinerary.poi_ids == ["CF-001", "CF-009", "CF-005", "CF-006", "CF-007"]
    assert result.adaptation.replaced == ["CF-003"] and result.validation.ok


def test_revise_itinerary_asks_for_an_itinerary_then_applies_new_preferences(itineraries, history_walk):
    assert itineraries.revise_itinerary(None, Preferences()).kind == "need_info"
    result = itineraries.revise_itinerary(history_walk, Preferences(interests=["history"], setting="outdoor"))
    assert result.adaptation.changed and result.validation.ok  # CF-007 is indoor


def test_report_weather_without_a_plan_is_the_forecast(itineraries):
    scenario = window(WED, "13:00", "17:00")
    hours = FixtureWeather().hourly(scenario.start, scenario.end)
    result = itineraries.report_weather(hours, scenario.span, "fixture", None, Preferences())
    assert result.rain_expected and result.itinerary is None and result.adaptation is None


def test_report_weather_adapts_the_plan_to_the_rain(itineraries):
    prefs = Preferences(interests=["history"], setting="outdoor")
    scenario = window(WED, "13:00", "17:00")
    plan = itineraries.plan_itinerary(scenario, prefs).itinerary
    hours = FixtureWeather().hourly(scenario.start, scenario.end)
    result = itineraries.report_weather(hours, scenario.span, "fixture", plan, prefs)
    assert result.itinerary.poi_ids == ["CF-001", "CF-003", "CF-007"]
    assert result.adaptation.changed and result.validation.ok
