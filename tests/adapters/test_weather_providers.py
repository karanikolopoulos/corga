import httpx

from tests.conftest import WED
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
