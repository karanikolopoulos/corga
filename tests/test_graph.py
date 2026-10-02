from datetime import date, time

import pytest

from langchain_core.messages import HumanMessage

from corga.agent.graph import build_graph
from corga.agent.state import ParsedRequest
from corga.data.timeutil import at

TUE, WED = date(2026, 10, 6), date(2026, 10, 7)


@pytest.fixture
def graph():
    return build_graph()


def run(graph, deps, thread, text="..."):
    """One user turn on `thread`; returns the state after it."""
    config = {"configurable": {"thread_id": thread}}
    graph.invoke({"messages": [HumanMessage(text)]}, config, context=deps)
    return graph.get_state(config).values


def test_replace_second_stop_keeps_the_rest(graph, make_deps, store):
    deps = make_deps(
        [
            ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(15), interests=["history"]),
            ParsedRequest(intent="edit_plan", stop_index=2, with_child=True),
        ]
    )
    first = run(graph, deps, "b")["itinerary"]
    final = run(graph, deps, "b")
    it = final["itinerary"]
    assert first.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-007"]
    assert it.poi_ids == ["CF-001", "CF-009", "CF-005", "CF-006", "CF-007"]
    assert store.get("CF-009").family_suitable is True
    assert it.stops[-1].end <= at(TUE, "15:00")
    assert final["preferences"].interests == ["history"] and final["result"]["validation"]["ok"]


def test_named_places_are_infeasible_with_alternative(graph, make_deps):
    names = [
        "Old Fortress",
        "New Fortress",
        "Achilleion Palace",
        "Mon Repos Museum of Palaiopolis",
        "Museum of Asian Art",
    ]
    request = ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(12, 30), requested_places=names)
    state = run(graph, make_deps([request]), "c")
    feasibility = state["result"]["feasibility"]
    found = {(i["code"], i["poi_id"]) for i in feasibility["issues"]}
    assert feasibility["feasible"] is False
    assert {("TIME_BUDGET", None), ("CLOSED_DAY", "CF-002"), ("LEG_TOO_LONG", "CF-021")} <= found
    assert state["itinerary"].poi_ids == ["CF-001"]


def test_weather_fixture_replans_within_window(graph, make_deps):
    deps = make_deps(
        [
            ParsedRequest(
                intent="plan",
                date=WED,
                start_time=time(13),
                end_time=time(17),
                interests=["history"],
                outdoor_only=True,
            ),
            ParsedRequest(intent="weather", date=WED, start_time=time(13), end_time=time(17)),
        ]
    )
    assert run(graph, deps, "d")["itinerary"].poi_ids == ["CF-002", "CF-003", "CF-005", "CF-006", "CF-009"]
    final = run(graph, deps, "d")
    assert final["result"]["source"] == "fixture" and final["result"]["adaptation"]["changed"]
    assert final["itinerary"].poi_ids == ["CF-002", "CF-003", "CF-007", "CF-017"]
    assert (final["window"].start, final["window"].end) == (at(WED, "13:00"), at(WED, "17:00"))


def test_refusals(graph, make_deps):
    deps = make_deps(
        [
            ParsedRequest(intent="plan", wants_fabrication=True),
            ParsedRequest(
                intent="plan", date=WED, start_time=time(13), end_time=time(16), requested_places=["Corfu Aquarium"]
            ),
        ]
    )
    assert run(graph, deps, "h")["result"] == {"kind": "refusal", "reason": "fabrication_or_ignore_dataset"}
    second = run(graph, deps, "h")["result"]
    assert second["reason"] == "not_in_dataset" and second["names"] == ["Corfu Aquarium"]


def test_unsupported_facts_are_flagged_not_answered(graph, make_deps):
    question = "Can I buy tickets for the Achilleion online, and when is Saint Spyridon open?"
    request = ParsedRequest(
        intent="question", question=question, requested_places=["Achilleion Palace", "Saint Spyridon Church"]
    )
    result = run(graph, make_deps([request], hit_ids=("CF-021",)), "g")["result"]
    facts = {f["id"]: f for f in result["facts"]}
    assert "tickets" in result["unsupported_topics"]
    assert facts["CF-004"]["opening_hours"] is None and "CF-021" in result["citations"]


def test_reply_contains_rendered_plan_and_sources(graph, make_deps):
    request = ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(15), interests=["history"])
    reply = run(graph, make_deps([request]), "r")["messages"][-1].content
    assert reply.startswith("Here is what I found.")
    assert "| 1 | Old Fortress (CF-001) |" in reply and "Sources (dataset records)" in reply


def test_missing_window_asks_for_it(graph, make_deps):
    deps = make_deps([ParsedRequest(intent="plan", interests=["history"])])
    assert run(graph, deps, "w")["result"]["kind"] == "need_info"


def test_threads_are_isolated(graph, make_deps):
    deps = make_deps(
        [
            ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(15), interests=["history"]),
            ParsedRequest(intent="edit_plan", stop_index=2),
        ]
    )
    run(graph, deps, "one")
    assert run(graph, deps, "two")["result"] == {"kind": "need_info", "missing": "an existing itinerary to edit"}
