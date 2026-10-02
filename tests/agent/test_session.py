from datetime import time

import pytest

from langchain_core.retrievers import BaseRetriever

from corga.tracing import JsonlTracer
from tests.conftest import WED, read, of_kind
from corga.agent.state import ParsedRequest
from corga.agent.runtime import ChatSession, build_graph

QUESTION = "Tell me about the Old Fortress"


class RaisingRetriever(BaseRetriever):
    """A retriever whose search raises the given exception."""

    exc: BaseException

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query, *, run_manager):
        raise self.exc

    def closest(self, query):
        return {doc.metadata["id"]: doc.metadata["score"] for doc in self.invoke(query)}

    def rank(self, query, allowed_ids):
        return []


def test_session_writes_a_line_per_node_retrieval_and_turn(make_deps, tmp_path):
    path = tmp_path / "sessions" / "abc.jsonl"
    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)])
    session = ChatSession(build_graph(), deps, JsonlTracer(path), session_id="abc")
    reply = session.send(QUESTION)
    session.close()
    events = read(path)
    assert events[0] == events[0] | {"kind": "session_started", "session_id": "abc"}
    assert events[-1]["kind"] == "session_closed"
    assert [e["node"] for e in of_kind(events, "node")] == ["parse", "answer", "respond"]
    [retrieval] = of_kind(events, "retrieval")
    assert (retrieval["node"], retrieval["query"]) == ("answer", QUESTION)
    assert retrieval["documents"] == [{"id": "CF-001", "name": "Old Fortress", "score": 1.0}]
    [turn] = of_kind(events, "turn")
    assert (turn["input"], turn["reply"], turn["error"]) == (QUESTION, reply, None)
    assert "CF-001 Old Fortress" in reply


def test_a_node_line_has_no_input_and_a_step_line_has_its_node_and_arguments(make_deps, tmp_path):
    deps = make_deps([ParsedRequest(intent="plan", date=WED, start_time=time(13), end_time=time(17))])
    session = ChatSession(build_graph(), deps, JsonlTracer(tmp_path / "s.jsonl"))
    session.send("plan")
    session.close()
    events = read(tmp_path / "s.jsonl")
    nodes = {e["node"]: e for e in of_kind(events, "node")}
    steps = {e["name"]: e for e in of_kind(events, "step")}
    assert "input" not in nodes["plan"] and nodes["plan"]["triggers"] == ["branch:to:plan"]
    assert list(steps) == ["select_candidates", "build_itinerary", "validate"]
    assert steps["build_itinerary"]["node"] == "plan" and set(steps["build_itinerary"]["input"]) == {"window", "pois"}
    assert nodes["plan"]["step"] == steps["build_itinerary"]["step"]  # the step runs in its node's graph step


def test_an_error_is_recorded_on_the_failing_run_and_its_node_and_turn_then_raised(make_deps, tmp_path):
    failing = RaisingRetriever(exc=RuntimeError("index unavailable"))
    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)], retriever=failing)
    session = ChatSession(build_graph(), deps, JsonlTracer(tmp_path / "s.jsonl"))
    with pytest.raises(RuntimeError, match="index unavailable"):
        session.send(QUESTION)
    session.close()
    error = "RuntimeError('index unavailable')"  # the error line only, no traceback
    failed = [(e["kind"], e.get("node")) for e in read(tmp_path / "s.jsonl") if e.get("error") == error]
    assert failed == [("retrieval", "answer"), ("node", "answer"), ("turn", None)]


def test_untraced_session_runs_and_writes_nothing(make_deps):
    session = ChatSession(build_graph(), make_deps([ParsedRequest(intent="question", question=QUESTION)]), None)
    assert session.trace_path is None and "Old Fortress" in session.send(QUESTION)
    assert "time window: none" in session.summary()
    session.close()


def test_ctrl_c_during_a_turn_is_traced_and_re_raised(make_deps, tmp_path):
    failing = RaisingRetriever(exc=KeyboardInterrupt())
    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)], retriever=failing)
    session = ChatSession(build_graph(), deps, JsonlTracer(tmp_path / "s.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        session.send(QUESTION)
    session.close()
    assert "turn_cancelled" in [e["kind"] for e in read(tmp_path / "s.jsonl")]


def test_a_weather_turn_records_the_tool_call_and_the_planning_steps_with_their_node(make_deps, tmp_path):
    deps = make_deps(
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
    session = ChatSession(build_graph(), deps, JsonlTracer(tmp_path / "s.jsonl"))
    session.send("plan")
    session.send("weather")
    session.close()
    events = read(tmp_path / "s.jsonl")
    [tool] = of_kind(events, "tool")
    assert (tool["node"], tool["name"], len(tool["output"])) == ("weather", "weather_forecast", 4)  # 13:00-16:00
    assert tool["input"] == {"start": "2026-10-07T13:00:00+03:00", "end": "2026-10-07T17:00:00+03:00"}
    steps = [(e["node"], e["name"]) for e in of_kind(events, "step")]
    assert steps[-3:] == [("weather", "select_candidates"), ("weather", "replan_for_rain"), ("weather", "validate")]
    assert [e["node"] for e in of_kind(events, "node")][-2:] == ["weather", "respond"]  # in finish order


def test_planning_steps_leave_nothing_in_the_checkpointer(make_deps):
    deps = make_deps([ParsedRequest(intent="plan", date=WED, start_time=time(13), end_time=time(17))])
    graph = build_graph()
    ChatSession(graph, deps, None).send("plan")
    writes = [write for task_writes in graph.checkpointer.writes.values() for write in task_writes.values()]
    channels = {channel for _, channel, _, _ in writes}
    assert "itinerary" in channels and "__return__" not in channels  # state only, no step results


def test_the_parser_is_built_once_and_reused_across_turns(make_deps):
    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)] * 2)
    session = ChatSession(build_graph(), deps, None)
    session.send("first")
    parser = deps.parser
    session.send("second")
    assert deps.parser is parser
