import pytest

from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

from tests.conftest import WED, ScriptedChatModel, call, window, itinerary
from corga.agent.state import TurnState
from corga.agent.ask_agent import MODEL_CALLS
from corga.agent.nodes.ask import ask
from corga.agent.runtime.deps import Deps
from corga.domain.planning.models import Itinerary, Preferences


def scripted(*messages) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))


@pytest.fixture
def plan_d(store):
    return itinerary(store, ["CF-001", "CF-003", "CF-005"], window(WED, "13:00", "17:00"))


def stored(plan: Itinerary) -> dict:
    """Pass the stored window, plan and preferences as the ask node does."""
    return {"window": plan.window, "itinerary": plan, "preferences": Preferences()}


def asked(deps: Deps, text: str, **state) -> dict:
    return ask(TurnState(messages=[HumanMessage(text)], **state), Runtime(context=deps))


def test_a_forecast_for_a_stop_reads_the_stored_plan(make_deps, plan_d):
    llm = scripted(call("forecast", stop=1), AIMessage("Dry."))
    deps = make_deps(llm=llm)
    out = deps.agent.invoke({"messages": [HumanMessage("Rain at stop 1?")], **stored(plan_d)}, context=deps)
    [result] = [m for m in out["messages"] if isinstance(m, ToolMessage)]

    assert result.text.startswith("[stop:1] Old Fortress 13:00-") and "[forecast:fixture:" in result.text


def test_a_tool_error_reaches_the_model_as_an_error_message(make_deps, plan_d):
    llm = scripted(call("forecast", stop=9), AIMessage("No such stop."))
    deps = make_deps(llm=llm)
    out = deps.agent.invoke({"messages": [HumanMessage("Rain at stop 9?")], **stored(plan_d)}, context=deps)
    [error] = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]

    assert error.text == "forecast failed: the plan has stops 1-3"


def test_the_model_call_limit_stops_a_run_that_never_answers(make_deps):
    llm = scripted(*(call("search", query="Liston") for _ in range(MODEL_CALLS)))

    with pytest.raises(ModelCallLimitExceededError):  # the graph's error handler asks to rephrase
        asked(make_deps(llm=llm), "What is the Liston?")
