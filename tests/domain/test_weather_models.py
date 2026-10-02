from tests.conftest import WED
from corga.domain.weather import HourlyWeather
from corga.domain.timeutil import TZ, at


def test_wmo_descriptions():
    assert HourlyWeather(time=at(WED, "15:00"), weather_code=63).description == "moderate rain"
    assert HourlyWeather(time=at(WED, "15:00")).description == "unknown"


def test_hourly_weather_is_always_athens_time():
    wall_clock = HourlyWeather(time="2026-10-07T15:00")  # fixture and Open-Meteo style: no offset
    utc = HourlyWeather(time="2026-10-07T12:00:00+00:00")
    assert wall_clock.time.tzinfo == TZ and wall_clock.time.hour == 15
    assert utc.time == wall_clock.time and utc.time.tzinfo == TZ  # 12:00 UTC is 15:00 in Athens (EEST)
