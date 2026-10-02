"""Live forecast from Open-Meteo (no API key). Errors surface as WeatherUnavailable."""

from typing import Any
from datetime import datetime

import httpx

from corga.data.timeutil import TZ
from corga.weather.models import HourlyWeather, WeatherUnavailable, overlapping

URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = "temperature_2m,precipitation,weather_code"


class OpenMeteoWeather:
    source = "live"
    label = "live (Open-Meteo)"

    def __init__(self, latitude: float, longitude: float, timeout_s: float = 10.0, client: httpx.Client | None = None):
        self._lat, self._lon, self._timeout = latitude, longitude, timeout_s
        self._client = client or httpx.Client()

    def hourly(self, start: datetime, end: datetime) -> list[HourlyWeather]:
        """Forecast hours overlapping `start`-`end` (Europe/Athens)."""
        return overlapping(self._parse(self._fetch(start, end)), start, end, "Open-Meteo")

    def _fetch(self, start: datetime, end: datetime) -> dict[str, Any]:
        """One request covering the whole days of `start`-`end`; any failure becomes WeatherUnavailable."""
        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "hourly": HOURLY_FIELDS,
            "timezone": TZ.key,  # times come back as local wall-clock time
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        }
        try:
            response = self._client.get(URL, params=params, timeout=self._timeout)
        except httpx.HTTPError as exc:  # network error or timeout
            raise WeatherUnavailable(f"Open-Meteo request failed: {exc}") from exc
        if response.is_error:  # 4xx / 5xx, e.g. a date outside the forecast range
            raise WeatherUnavailable(f"Open-Meteo error {response.status_code}: {_reason(response)}")
        return response.json()

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[HourlyWeather]:
        """Open-Meteo returns one list per field; zip them back into hours."""
        hourly = payload["hourly"]
        columns = zip(hourly["time"], hourly["temperature_2m"], hourly["precipitation"], hourly["weather_code"])
        return [
            HourlyWeather(
                time=time,
                temperature_c=temperature,
                precipitation_mm=precipitation or 0.0,  # missing values come back as null
                weather_code=code,
            )
            for time, temperature, precipitation, code in columns
        ]


def _reason(response: httpx.Response) -> str:
    """Open-Meteo explains errors in a JSON `reason` field; fall back to the raw body."""
    try:
        return response.json().get("reason", response.text)
    except ValueError:
        return response.text
