import json
import logging

from pathlib import Path
from datetime import datetime

from corga import RAIN_FIXTURE_PATH
from corga.domain.weather import HourlyWeather, overlapping

log = logging.getLogger(__name__)


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
        found = overlapping(self._hours, start, end, "fixture")

        log.debug("fixture: %d hours for %s", len(found), f"{start:%a %H:%M}-{end:%H:%M}")

        return found
