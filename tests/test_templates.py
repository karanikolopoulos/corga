from tests.conftest import WED, RULES, window
from corga.agent.templates import render_result
from corga.planning.models import Itinerary
from corga.planning.schedule import Schedule


def test_plan_result_renders_table_assumptions_and_sources(store):
    scenario = window(WED, "13:00", "17:00")
    stops = list(Schedule.of([store.get("CF-007"), store.get("CF-012")], scenario.start, RULES.walking))
    it = Itinerary(window=scenario, stops=stops, assumptions=RULES.assumptions())
    text = render_result({"kind": "plan", "itinerary": it.model_dump(mode="json"), "citations": it.poi_ids}, store)
    assert "| 1 | Casa Parlante Museum (CF-007) | – | 13:00 | 13:00–13:55 | indoor |" in text
    assert "| 2 | Corfu Philharmonic Music Museum (CF-012) | 1 min | 13:56 |" in text
    assert "*Assumptions:*" in text and "**Sources (dataset records):** CF-007 Casa Parlante Museum" in text


def test_infeasible_result_lists_reasons_first(store):
    result = {
        "kind": "plan",
        "feasibility": {"feasible": False, "issues": [{"message": "Too far."}], "alternative": None},
    }
    assert render_result(result, store).startswith("**Why this does not fit:**\n- Too far.")


def test_forecast_and_recommendations_render_as_lists(store):
    result = {
        "kind": "weather",
        "source": "fixture",
        "hours": [{"time": "Wed 15:00", "precipitation_mm": 2.1, "description": "rain", "temperature_c": 19.0}],
        "items": [{"id": "CF-001", "name": "Old Fortress", "category": "history", "setting": "outdoor"}],
    }
    text = render_result(result, store)
    assert "**Forecast (fixture)**" in text and "| Wed 15:00 | 2.1 | rain | 19.0 |" in text
    assert "- Old Fortress (CF-001): history, outdoor" in text
