"""Deterministic rain handling: exposed outdoor/mixed stops are replaced by indoor ones."""

from collections.abc import Sequence

from pydantic import BaseModel

from corga.data.store import POIStore
from corga.data.models import POI
from corga.data.timeutil import Interval
from corga.planning.rules import PlanningRules
from corga.weather.models import HourlyWeather
from corga.planning.models import Stop, Itinerary
from corga.planning.builder import build_itinerary


class WeatherAdaptation(BaseModel):
    changed: bool
    wet_periods: list[Interval]
    replaced: list[str] = []
    itinerary: Itinerary
    note: str


def wet_periods(hours: Sequence[HourlyWeather]) -> list[Interval]:
    """Wet hours in time order, with back-to-back hours merged (15:00 and 16:00 become 15:00-17:00)."""
    periods: list[Interval] = []
    for hour in sorted(hours, key=lambda h: h.time):
        if not hour.is_wet:
            continue
        if periods and periods[-1].end == hour.span.start:  # continues the previous wet period
            periods[-1] = periods[-1]._replace(end=hour.span.end)
        else:
            periods.append(hour.span)
    return periods


def is_exposed(stop: Stop, periods: Sequence[Interval]) -> bool:
    """An outdoor or mixed stop (mixed is not shelter) visited while it rains."""
    return stop.setting in ("outdoor", "mixed") and any(period.overlaps(stop.span) for period in periods)


def adapt_for_weather(
    itinerary: Itinerary,
    hours: Sequence[HourlyWeather],
    store: POIStore,
    candidates: Sequence[POI],
    rules: PlanningRules,
) -> WeatherAdaptation:
    periods = wet_periods(hours)
    exposed = [i for i, stop in enumerate(itinerary.stops) if is_exposed(stop, periods)]
    if not exposed:
        note = "No outdoor or mixed stop overlaps forecast rain."
        return WeatherAdaptation(changed=False, wet_periods=periods, itinerary=itinerary, note=note)

    # Keep every stop before the first exposed one; rebuild the rest of the window from indoor places only.
    kept = [store.get(stop.poi_id) for stop in itinerary.stops[: exposed[0]]]
    indoor = [poi for poi in candidates if poi.setting == "indoor"]
    new = build_itinerary(itinerary.window, indoor, rules, prefix=kept)

    replaced = [pid for pid in itinerary.poi_ids if pid not in new.poi_ids]
    note = f"Rain expected from {periods[0].start:%H:%M}; replaced {len(replaced)} exposed stop(s) with indoor places."
    return WeatherAdaptation(changed=True, wet_periods=periods, replaced=replaced, itinerary=new, note=note)
