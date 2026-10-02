from datetime import date

import httpx
import pytest

from tests.conftest import WED
from corga.domain.weather import WeatherUnavailable
from corga.domain.timeutil import at
from corga.adapters.weather.fixture import FixtureWeather
from corga.adapters.weather.open_meteo import OpenMeteoWeather

RECORDED = {
    "hourly": {
        "time": ["2026-10-07T13:00", "2026-10-07T14:00", "2026-10-07T15:00"],
        "temperature_2m": [23.1, 22.8, 21.0],
        "precipitation": [0.0, 0.0, 2.4],
        "weather_code": [1, 3, 63],
    }
}


def test_fixture_returns_overlapping_local_hours():
    hours = FixtureWeather().hourly(at(WED, "13:00"), at(WED, "17:00"))
    assert [h.time.hour for h in hours] == [13, 14, 15, 16]
    assert hours[0].time.tzinfo is not None
    assert [h.is_wet for h in hours] == [False, False, True, True]


def test_fixture_outside_its_date_is_unavailable():
    with pytest.raises(WeatherUnavailable):
        FixtureWeather().hourly(at(date(2026, 10, 8), "13:00"), at(date(2026, 10, 8), "17:00"))


def test_open_meteo_request_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=RECORDED)

    provider = OpenMeteoWeather(39.62, 19.92, client=httpx.Client(transport=httpx.MockTransport(handler)))
    hours = provider.hourly(at(WED, "13:00"), at(WED, "16:00"))
    assert seen["timezone"] == "Europe/Athens" and seen["start_date"] == "2026-10-07"
    assert "precipitation" in seen["hourly"] and "weather_code" in seen["hourly"]
    assert [h.precipitation_mm for h in hours] == [0.0, 0.0, 2.4]
    assert provider.source == "live"


def test_open_meteo_errors_become_weather_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": True, "reason": "Parameter 'start_date' is out of allowed range"})

    provider = OpenMeteoWeather(39.62, 19.92, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(WeatherUnavailable, match="out of allowed range"):
        provider.hourly(at(WED, "13:00"), at(WED, "16:00"))


def test_open_meteo_without_hours_in_the_range_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RECORDED)  # Wednesday's hours only

    provider = OpenMeteoWeather(39.62, 19.92, client=httpx.Client(transport=httpx.MockTransport(handler)))
    thursday = date(2026, 10, 8)
    with pytest.raises(WeatherUnavailable, match="Open-Meteo has no data for 2026-10-08 13:00-16:00"):
        provider.hourly(at(thursday, "13:00"), at(thursday, "16:00"))


@pytest.mark.integration
def test_live_open_meteo_next_24_hours():
    from datetime import timedelta

    from corga.domain.timeutil import now

    start = now().replace(minute=0, second=0, microsecond=0)
    hours = OpenMeteoWeather(39.6243, 19.9217).hourly(start, start + timedelta(hours=24))
    assert len(hours) == 24


def test_providers_say_where_the_forecast_comes_from():
    assert FixtureWeather().label == "fixture (synthetic rain, corfu_synthetic_rain.json)"
    assert OpenMeteoWeather(39.62, 19.92).label == "live (Open-Meteo)"
