from tests.conftest import WED, window, itinerary
from corga.agent.results import Item, PlanResult, WeatherResult, RecommendationResult
from corga.data.timeutil import at
from corga.weather.models import HourlyWeather
from corga.agent.templates import render_result
from corga.planning.models import Violation, Preferences
from corga.planning.builder import FeasibilityReport


def test_plan_result_renders_table_assumptions_and_sources(store):
    it = itinerary(store, ["CF-007", "CF-012"], window(WED, "13:00", "17:00"))
    text = render_result(PlanResult(itinerary=it))
    assert "| 1 | Casa Parlante Museum (CF-007) | – | 13:00 | 13:00–13:55 | indoor |" in text
    assert "| 2 | Corfu Philharmonic Music Museum (CF-012) | 1 min | 13:56 |" in text
    assert "*Assumptions:*" in text
    sources = "**Sources (dataset records):** CF-007 Casa Parlante Museum; CF-012 Corfu Philharmonic Music Museum"
    assert text.endswith(sources)


def test_infeasible_result_lists_reasons_first():
    issue = Violation(code="TIME_BUDGET", message="Too far.")
    result = PlanResult(feasibility=FeasibilityReport(available_minutes=90, visit_minutes=300, issues=[issue]))
    assert render_result(result).startswith("**Why this does not fit:**\n- Too far.")


def test_forecast_renders_as_a_table():
    hour = HourlyWeather(time=at(WED, "15:00"), precipitation_mm=2.1, temperature_c=19.0, weather_code=61)
    result = WeatherResult(source="fixture", range=window(WED, "13:00", "17:00").span, wet_periods=[], hours=[hour])
    text = render_result(result)
    assert "**Forecast (fixture)**" in text and "| Wed 15:00 | 2.1 | light rain | 19.0 |" in text


def test_recommendations_render_as_a_list(store):
    item = Item.of(store.get("CF-001"), Preferences())
    text = render_result(RecommendationResult(preferences=Preferences(), items=[item]))
    assert "- Old Fortress (CF-001): history, mixed" in text
