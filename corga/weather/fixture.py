"""Synthetic weather injected for reproducible replanning tests."""

import json

from pathlib import Path
from datetime import datetime

from corga.data.timeutil import TZ, Interval
from corga.weather.models import HourlyWeather, WeatherUnavailable


class FixtureWeather:
    source = "fixture"

    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._hours = [
            HourlyWeather(
                time=datetime.fromisoformat(h["local_time"]).replace(tzinfo=TZ),  # local_time is Europe/Athens
                temperature_c=h["temperature_c"],
                precipitation_mm=h["precipitation_mm"],
                weather_code=h["weather_code"],
            )
            for h in raw["hourly"]
        ]

    def hourly(self, start: datetime, end: datetime) -> list[HourlyWeather]:
        hours = [h for h in self._hours if h.span.overlaps(Interval(start, end))]
        if not hours:
            raise WeatherUnavailable(f"fixture has no data for {start:%Y-%m-%d %H:%M}-{end:%H:%M}")
        return hours
