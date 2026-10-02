from tests.conftest import WED
from corga.agent.tools import weather_forecast
from corga.domain.timeutil import at
from corga.adapters.weather.fixture import FixtureWeather


def test_the_weather_tool_takes_its_provider_from_the_caller_only():
    assert sorted(weather_forecast.tool_call_schema.model_json_schema()["properties"]) == ["end", "start"]
    hours = weather_forecast.invoke({"start": at(WED, "13:00"), "end": at(WED, "17:00"), "provider": FixtureWeather()})
    assert [hour.time.hour for hour in hours] == [13, 14, 15, 16]
