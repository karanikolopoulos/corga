"""Synthetic weather injected for reproducible replanning tests."""

import json

from pathlib import Path
from datetime import datetime

from corga import RAIN_FIXTURE_PATH
from corga.data.timeutil import Interval
from corga.weather.models import HourlyWeather, WeatherUnavailable


class FixtureWeather:
    source = "fixture"

    def __init__(self, path: str | Path = RAIN_FIXTURE_PATH):
        self.label = f"fixture (synthetic rain, {Path(path).name})"
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._hours = [
            HourlyWeather(
                time=h["local_time"],
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
