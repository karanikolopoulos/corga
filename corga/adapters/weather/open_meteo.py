import logging

from typing import Any
from datetime import datetime

import httpx

from corga.domain.weather import HourlyWeather, WeatherUnavailable, overlapping
from corga.domain.timeutil import TZ

log = logging.getLogger(__name__)

URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = "temperature_2m,precipitation,weather_code"


def _reason(response: httpx.Response) -> str:
    """Read why Open-Meteo refused a request."""
    try:
        return response.json().get("reason", response.text)
    except ValueError:
        return response.text


class OpenMeteoWeather:
    """The live Open-Meteo forecast for one location. Any failure raises `WeatherUnavailable`."""

    source = "live"
    label = "live (Open-Meteo)"

    def __init__(self, latitude: float, longitude: float, timeout_s: float = 10.0, client: httpx.Client | None = None):
        self._lat, self._lon, self._timeout = latitude, longitude, timeout_s
        self._client = client or httpx.Client()

    def hourly(self, start: datetime, end: datetime) -> list[HourlyWeather]:
        """Fetch the forecast hours in Europe/Athens time.

        Args:
            start: The first hour wanted.
            end: The end of the range.

        Returns:
            The hours overlapping `start`-`end`.

        Raises:
            WeatherUnavailable: The request fails or returns no hours.
        """
        return overlapping(self._parse(self._fetch(start, end)), start, end, "Open-Meteo")

    def _fetch(self, start: datetime, end: datetime) -> dict[str, Any]:
        """Request the whole days of `start`-`end` in one call."""
        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "hourly": HOURLY_FIELDS,
            "timezone": TZ.key,  # times come back as local wall-clock time
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        }

        log.debug("Open-Meteo request: %s", params)

        try:
            response = self._client.get(URL, params=params, timeout=self._timeout)
        except httpx.HTTPError as exc:  # network error or timeout
            log.warning("Open-Meteo failed: %s", exc)
            raise WeatherUnavailable(f"Open-Meteo request failed: {exc}") from exc

        if response.is_error:  # 4xx / 5xx, e.g. a date outside the forecast range
            log.warning("Open-Meteo failed: error %s", response.status_code)
            raise WeatherUnavailable(f"Open-Meteo error {response.status_code}: {_reason(response)}")

        return response.json()

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[HourlyWeather]:
        """Zip Open-Meteo's lists, one per field, back into hours."""
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
