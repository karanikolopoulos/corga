from typing import Protocol
from datetime import datetime, timedelta
from collections.abc import Sequence

from pydantic import BaseModel, computed_field, field_validator

from corga.domain.timeutil import TZ, Interval, iso_duration

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
        """Convert to Europe/Athens time. A time without an offset is read as Athens time."""
        return value.replace(tzinfo=TZ) if value.tzinfo is None else value.astimezone(TZ)

    @property
    def span(self) -> Interval:
        return Interval(self.time, self.time + HOUR)

    @property
    def is_wet(self) -> bool:
        """Measurable rain, or a WMO code for rain.

        Drizzle, rain, showers and thunderstorms count.
        """
        return self.precipitation_mm >= WET_THRESHOLD_MM or self.weather_code in RAIN_CODES

    @computed_field
    @property
    def description(self) -> str:
        if self.weather_code is None:
            return "unknown"

        return DESCRIPTIONS.get(self.weather_code, f"WMO code {self.weather_code}")


def wet_periods(hours: Sequence[HourlyWeather]) -> list[Interval]:
    """Merge the wet hours into periods, in time order.

    Args:
        hours: The forecast hours.

    Returns:
        The rain periods. 15:00 and 16:00 become 15:00-17:00.
    """
    periods: list[Interval] = []

    for hour in sorted(hours, key=lambda h: h.time):
        if not hour.is_wet:
            continue

        if periods and periods[-1].meets(hour.span):
            periods[-1] = periods[-1]._replace(end=hour.span.end)
        else:
            periods.append(hour.span)

    return periods


class WeatherReport(BaseModel):
    """A forecast as fetched: where from, the hours asked for, and the rain in them."""

    source: str  # "live" or "fixture"
    label: str  # for people: "live (Open-Meteo)"
    start: datetime
    end: datetime
    hours: list[HourlyWeather]

    @computed_field
    @property
    def wet_periods(self) -> list[Interval]:
        return wet_periods(self.hours)

    @computed_field
    @property
    def rain_expected(self) -> bool:
        return bool(self.wet_periods)

    @property
    def evidence_id(self) -> str:
        """How an answer cites this forecast: its source and its ISO 8601 interval, the start and the duration."""
        return f"forecast:{self.source}:{self.start:%Y-%m-%dT%H:%M}/{iso_duration(self.end - self.start)}"


class WeatherUnavailable(RuntimeError): ...


def overlapping(hours: Sequence[HourlyWeather], start: datetime, end: datetime, source: str) -> list[HourlyWeather]:
    """Keep the hours overlapping `start`-`end`.

    Args:
        hours: The forecast hours.
        start: The start of the range.
        end: The end of the range.
        source: The provider, for the error.

    Returns:
        The hours.

    Raises:
        WeatherUnavailable: No hour overlaps.
    """
    span = Interval(start, end)
    found = [hour for hour in hours if hour.span.overlaps(span)]
    if not found:
        raise WeatherUnavailable(f"{source} has no data for {start:%Y-%m-%d %H:%M}-{end:%H:%M}")

    return found


class WeatherProvider(Protocol):
    source: str  # "live" or "fixture", recorded in every weather result
    label: str  # where the forecast comes from, for people: "live (Open-Meteo)"

    def hourly(self, start: datetime, end: datetime) -> list[HourlyWeather]:
        """Fetch the forecast hours.

        Args:
            start: The start of the range.
            end: The end of the range.

        Returns:
            The hours overlapping `start`-`end`.

        Raises:
            WeatherUnavailable: The source fails or has no hours.
        """
        ...
