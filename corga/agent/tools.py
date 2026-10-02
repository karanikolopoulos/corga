"""The agent's LangChain tools. The weather node calls `weather_forecast`; the tool is not bound to the LLM."""

from typing import Annotated
from datetime import datetime

from langchain_core.tools import InjectedToolArg, tool

from corga.domain.weather import HourlyWeather, WeatherProvider


@tool
def weather_forecast(
    start: datetime, end: datetime, provider: Annotated[WeatherProvider, InjectedToolArg]
) -> list[HourlyWeather]:
    """Hourly forecast for Corfu from `start` to `end` (Europe/Athens)."""
    return provider.hourly(start, end)
