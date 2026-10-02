"""Deterministic rain handling: exposed outdoor/mixed stops are replaced by indoor ones."""

from collections.abc import Callable, Sequence

from corga.data.store import POIStore
from corga.data.models import POI
from corga.data.timeutil import Interval
from corga.planning.rules import PlanningRules
from corga.weather.models import HourlyWeather
from corga.planning.models import Stop, Itinerary
from corga.planning.builder import Replan, replan_from_first_misfit


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
    periods: Sequence[Interval],
    indoor_candidates: Callable[[], Sequence[POI]],
    store: POIStore,
    rules: PlanningRules,
) -> Replan:
    """The itinerary rebuilt from indoor places from its first rain-exposed stop onwards."""
    replan = replan_from_first_misfit(
        itinerary, lambda stop: not is_exposed(stop, periods), indoor_candidates, store, rules
    )
    if not replan.changed:
        return replan.model_copy(update={"note": "No outdoor or mixed stop overlaps forecast rain."})
    count = len(replan.replaced)
    note = f"Rain expected from {periods[0].start:%H:%M}; replaced {count} exposed stop(s) with indoor places."
    return replan.model_copy(update={"note": note})
