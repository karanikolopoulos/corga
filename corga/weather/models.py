from typing import Protocol
from datetime import datetime, timedelta

from pydantic import BaseModel, field_validator

from corga.data.timeutil import TZ, Interval

HOUR = timedelta(hours=1)
WET_THRESHOLD_MM = 0.1
RAIN_CODES = frozenset(  # WMO drizzle, rain, showers and thunderstorms
    {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
)
DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


class HourlyWeather(BaseModel):
    time: datetime
    temperature_c: float | None = None
    precipitation_mm: float = 0.0
    weather_code: int | None = None

    @field_validator("time")
    @classmethod
    def _local(cls, value: datetime) -> datetime:
        """Europe/Athens time: a time without an offset is read as Athens wall-clock time, others are converted."""
        return value.replace(tzinfo=TZ) if value.tzinfo is None else value.astimezone(TZ)

    @property
    def span(self) -> Interval:
        """The hour this forecast covers."""
        return Interval(self.time, self.time + HOUR)

    @property
    def is_wet(self) -> bool:
        """Measurable rain, or a WMO code for drizzle, rain, showers or thunderstorms."""
        return self.precipitation_mm >= WET_THRESHOLD_MM or self.weather_code in RAIN_CODES

    @property
    def description(self) -> str:
        if self.weather_code is None:
            return "unknown"
        return DESCRIPTIONS.get(self.weather_code, f"WMO code {self.weather_code}")


class WeatherUnavailable(RuntimeError):
    """The provider has no forecast for the requested range."""


class WeatherProvider(Protocol):
    source: str  # "live" or "fixture", recorded in every weather result
    label: str  # where the forecast comes from, for people: "live (Open-Meteo)"

    def hourly(self, start: datetime, end: datetime) -> list[HourlyWeather]: ...
