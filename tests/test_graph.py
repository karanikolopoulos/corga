"""The agent end to end through `ChatSession`, as the REPL and the eval drive it; the LLM is scripted."""

from datetime import time

import pytest

from tests.conftest import TUE, WED
from corga.agent.state import AgentState, ParsedRequest
from corga.agent.results import RefusalResult, NeedInfoResult
from corga.agent.runtime import ChatSession, build_graph
from corga.data.timeutil import at

TUESDAY_HISTORY = ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(15), interests=["history"])


@pytest.fixture
def chat(make_deps):
    """Start untraced chat sessions on one graph; each gets the scripted requests (or given deps)."""
    graph = build_graph()

    def start(requests=(), *, deps=None, session_id=None, **deps_options):
        return ChatSession(graph, deps or make_deps(requests, **deps_options), None, session_id=session_id)

    return start


def turn(session: ChatSession, text: str = "...") -> AgentState:
    """Send one message and return the session state after it."""
    session.send(text)
    return session.state()


def test_replace_second_stop_keeps_the_rest(chat, store):
    session = chat(
        [
            TUESDAY_HISTORY,
            ParsedRequest(intent="edit_plan", stop_index=2, with_child=True),
        ]
    )
    first = turn(session).itinerary
    final = turn(session)
    it = final.itinerary
    assert first.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-007"]
    assert it.poi_ids == ["CF-001", "CF-009", "CF-005", "CF-006", "CF-007"]
    assert store.get("CF-009").family_suitable is True
    assert it.stops[-1].end <= at(TUE, "15:00")
    assert final.preferences.interests == ["history"] and final.result.validation.ok
    assert final.result.adaptation.replaced == ["CF-003"]


def test_named_places_are_infeasible_with_alternative(chat):
    names = [
        "Old Fortress",
        "New Fortress",
        "Achilleion Palace",
        "Mon Repos Museum of Palaiopolis",
        "Museum of Asian Art",
    ]
    request = ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(12, 30), requested_places=names)
    state = turn(chat([request]))
    feasibility = state.result.feasibility
    found = {(i.code, i.poi_id) for i in feasibility.issues}
    assert feasibility.feasible is False
    assert {("TIME_BUDGET", None), ("CLOSED_DAY", "CF-002"), ("LEG_TOO_LONG", "CF-021")} <= found
    assert state.itinerary.poi_ids == ["CF-001"]


def test_weather_fixture_replans_within_window(chat):
    session = chat(
        [
            ParsedRequest(
                intent="plan",
                date=WED,
                start_time=time(13),
                end_time=time(17),
                interests=["history"],
                setting="outdoor",
            ),
            ParsedRequest(intent="weather", date=WED, start_time=time(13), end_time=time(17)),
        ]
    )
    assert turn(session).itinerary.poi_ids == ["CF-001", "CF-003", "CF-005", "CF-006", "CF-009"]
    final = turn(session)
    assert final.result.source == "fixture" and final.result.adaptation.changed
    assert final.itinerary.poi_ids == ["CF-001", "CF-003", "CF-007"]
    assert (final.window.start, final.window.end) == (at(WED, "13:00"), at(WED, "17:00"))


def test_refusals(chat):
    session = chat(
        [
            ParsedRequest(intent="plan", wants_fabrication=True),
            ParsedRequest(
                intent="plan", date=WED, start_time=time(13), end_time=time(16), requested_places=["Corfu Aquarium"]
            ),
        ]
    )
    assert turn(session).result == RefusalResult(reason="fabrication_or_ignore_dataset")
    second = turn(session).result
    assert second.reason == "not_in_dataset" and second.names == ["Corfu Aquarium"]


def test_unsupported_facts_are_flagged_not_answered(chat):
    question = "Can I buy tickets for the Achilleion online, and when is Saint Spyridon open?"
    request = ParsedRequest(
        intent="question", question=question, requested_places=["Achilleion Palace", "Saint Spyridon Church"]
    )
    result = turn(chat([request], hit_ids=("CF-021",))).result
    facts = {f.id: f for f in result.facts}
    assert "tickets" in result.unsupported_topics
    assert facts["CF-004"].opening_hours is None and "CF-021" in [c.id for c in result.citations]
    assert facts["CF-021"].score == 1.0  # returned by the (stub) search
    assert facts["CF-004"].score is None  # named by the user, not returned by the search


def test_reply_contains_rendered_plan_and_sources(chat):
    reply = chat([TUESDAY_HISTORY]).send("...")
    assert reply.startswith("Here is what I found.")
    assert "| 1 | Old Fortress (CF-001) |" in reply and "Sources (dataset records)" in reply


def test_missing_window_asks_for_it(chat):
    assert turn(chat([ParsedRequest(intent="plan", interests=["history"])])).result.kind == "need_info"


def test_sessions_are_isolated(chat, make_deps):
    deps = make_deps(
        [
            TUESDAY_HISTORY,
            ParsedRequest(intent="edit_plan", stop_index=2),
        ]
    )
    turn(chat(deps=deps, session_id="one"))
    second = turn(chat(deps=deps, session_id="two"))  # same graph and deps, another thread
    assert second.result == NeedInfoResult(missing="an existing itinerary to edit")


def test_a_whole_plan_edit_applies_the_new_setting_and_keeps_fitting_stops(chat, store):
    session = chat(
        [
            ParsedRequest(
                intent="plan", date=TUE, start_time=time(11), end_time=time(15), interests=["history"], setting="indoor"
            ),
            ParsedRequest(intent="edit_plan", setting="outdoor"),  # "make it outdoor instead": no stop number
        ]
    )
    indoor = turn(session).itinerary
    assert {store.get(pid).setting for pid in indoor.poi_ids} == {"indoor"}
    final = turn(session)
    assert {store.get(pid).setting for pid in final.itinerary.poi_ids} <= {"outdoor", "mixed"}
    assert final.preferences.setting == "outdoor" and final.result.adaptation.changed
    assert final.window == indoor.window and final.result.validation.ok


def test_an_edit_with_neither_a_stop_nor_a_preference_asks_which_stop(chat):
    session = chat(
        [
            TUESDAY_HISTORY,
            ParsedRequest(intent="edit_plan"),
        ]
    )
    turn(session)
    assert turn(session).result.kind == "need_info"


def test_closing_a_session_frees_its_checkpoints_only(chat):
    closed, kept = chat([TUESDAY_HISTORY], session_id="closed"), chat([TUESDAY_HISTORY], session_id="kept")
    turn(closed)
    turn(kept)
    closed.close()
    assert "closed" not in closed.graph.checkpointer.storage
    assert closed.state().itinerary is None and kept.state().itinerary is not None
