import pytest

from jinja2 import UndefinedError

from tests.conftest import WED, window, itinerary
from corga.agent.state import AgentState
from corga.agent.prompts import PromptStore
from corga.domain.weather import HourlyWeather
from corga.domain.timeutil import at
from corga.services.results import Item, PlanResult, WeatherResult, RecommendationResult
from corga.domain.planning.models import Violation, Preferences
from corga.domain.planning.builder import FeasibilityReport

PROMPTS = PromptStore()


def reply(result) -> str:
    return PROMPTS["reply"].render(result=result).strip()


def test_the_store_lists_the_templates_by_name():
    assert sorted(PROMPTS.names) == ["parse", "reply", "respond", "state"]


def test_an_unknown_template_names_the_available_ones():
    with pytest.raises(KeyError, match="parse, reply, respond, state"):
        PROMPTS["parse_terse"]


def test_a_missing_value_is_an_error():
    with pytest.raises(UndefinedError):
        PROMPTS["parse"].render(state=AgentState())  # no `today`


def test_the_parse_prompt_carries_the_date_and_the_state():
    text = PROMPTS["parse"].render(today="2026-10-06", state=AgentState())
    assert "Today is 2026-10-06 (Europe/Athens)." in text
    assert "time window: none\npreferences: none\ncurrent itinerary: none" in text


def test_plan_result_renders_table_assumptions_and_sources(store):
    it = itinerary(store, ["CF-007", "CF-012"], window(WED, "13:00", "17:00"))
    text = reply(PlanResult(itinerary=it))
    assert "| 1 | Casa Parlante Museum (CF-007) | – | 13:00 | 13:00–13:55 | indoor |" in text
    assert "| 2 | Corfu Philharmonic Music Museum (CF-012) | 1 min | 13:56 |" in text
    assert "*Assumptions:*" in text
    sources = "**Sources (dataset records):** CF-007 Casa Parlante Museum; CF-012 Corfu Philharmonic Music Museum"
    assert text.endswith(sources)


def test_infeasible_result_lists_reasons_first():
    issue = Violation(code="TIME_BUDGET", message="Too far.")
    result = PlanResult(feasibility=FeasibilityReport(available_minutes=90, visit_minutes=300, issues=[issue]))
    assert reply(result).startswith("**Why this does not fit:**\n- Too far.")


def test_forecast_renders_as_a_table():
    hour = HourlyWeather(time=at(WED, "15:00"), precipitation_mm=2.1, temperature_c=19.0, weather_code=61)
    result = WeatherResult(source="fixture", range=window(WED, "13:00", "17:00").span, wet_periods=[], hours=[hour])
    text = reply(result)
    assert "**Forecast (fixture)**" in text and "| Wed 15:00 | 2.1 | light rain | 19.0 |" in text


def test_recommendations_render_as_a_list(store):
    item = Item.of(store.get("CF-001"), Preferences())
    text = reply(RecommendationResult(preferences=Preferences(), items=[item]))
    assert "- Old Fortress (CF-001): history, mixed" in text
