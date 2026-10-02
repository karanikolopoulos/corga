import json

import pytest

from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from tests.conftest import WED
from corga.agent.graph import build_graph
from corga.agent.state import ParsedRequest
from corga.agent.session import ChatSession
from corga.agent.tracing import JsonlTracer
from corga.data.timeutil import at
from corga.planning.models import TimeWindow

QUESTION = "Tell me about the Old Fortress"


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_session_traces_nodes_and_the_turn(make_deps, tmp_path):
    path = tmp_path / "sessions" / "abc.jsonl"
    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)])
    session = ChatSession(build_graph(), deps, JsonlTracer(path), session_id="abc")
    reply = session.send(QUESTION)
    session.close()
    events = read(path)
    assert events[0] == events[0] | {"kind": "session_started", "session_id": "abc"}
    assert events[-1]["kind"] == "session_closed"
    assert [e["node"] for e in events if e["kind"] == "node"] == ["parse", "answer", "respond"]
    turn = next(e for e in events if e["kind"] == "turn")
    assert turn["input"] == [{"type": "human", "text": QUESTION}] and turn["reply"]["text"] == reply
    assert "CF-001 Old Fortress" in reply


def test_llm_calls_record_model_latency_and_usage(tmp_path):
    usage = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
    llm = GenericFakeChatModel(messages=iter([AIMessage("hi", usage_metadata=usage)]))
    tracer = JsonlTracer(tmp_path / "s.jsonl")
    llm.invoke("hello", config={"callbacks": [tracer]})
    tracer.close()
    call = read(tmp_path / "s.jsonl")[0]
    assert call["kind"] == "llm_call" and call["usage"] == usage and call["latency_s"] >= 0


def test_node_errors_are_traced_and_raised(make_deps, store, tmp_path):
    class BrokenRetriever:
        def search(self, *args, **kwargs):
            raise RuntimeError("index unavailable")

        def rank(self, *args, **kwargs):
            return []

    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)])
    deps.retriever = BrokenRetriever()
    session = ChatSession(build_graph(), deps, JsonlTracer(tmp_path / "s.jsonl"))
    with pytest.raises(RuntimeError, match="index unavailable"):
        session.send(QUESTION)
    session.close()
    errors = [e for e in read(tmp_path / "s.jsonl") if e["kind"] == "error"]
    line = "RuntimeError('index unavailable')"  # the error line only, no traceback
    assert [(e["node"], e.get("step"), e["error"]) for e in errors] == [
        ("answer", "retrieve", line),
        ("answer", None, line),
    ]


def test_close_is_idempotent(tmp_path):
    tracer = JsonlTracer(tmp_path / "s.jsonl")
    tracer.close()
    tracer.close()
    assert [e["kind"] for e in read(tmp_path / "s.jsonl")] == ["session_closed"]


def test_tracer_serialises_pydantic_and_datetimes(tmp_path):
    tracer = JsonlTracer(tmp_path / "s.jsonl")
    day = at(WED, "13:00")
    tracer.event("probe", window=TimeWindow(start=day, end=day.replace(hour=17)))
    tracer.close()
    assert read(tmp_path / "s.jsonl")[0]["window"]["start"] == "2026-10-07T13:00:00+03:00"


def test_untraced_session_runs_and_writes_nothing(make_deps, tmp_path):
    session = ChatSession(build_graph(), make_deps([ParsedRequest(intent="question", question=QUESTION)]), None)
    assert session.trace_path is None and "Old Fortress" in session.send(QUESTION)
    assert "time window: none" in session.summary()
    session.close()


def test_ctrl_c_during_a_turn_is_traced_and_re_raised(make_deps, tmp_path):
    class Interrupting:
        def search(self, *args, **kwargs):
            raise KeyboardInterrupt

        def rank(self, *args, **kwargs):
            return []

    deps = make_deps([ParsedRequest(intent="question", question=QUESTION)])
    deps.retriever = Interrupting()
    session = ChatSession(build_graph(), deps, JsonlTracer(tmp_path / "s.jsonl"))
    with pytest.raises(KeyboardInterrupt):
        session.send(QUESTION)
    session.close()
    assert "turn_cancelled" in [e["kind"] for e in read(tmp_path / "s.jsonl")]


def test_a_weather_turn_records_the_tool_call_and_the_planning_steps(make_deps, tmp_path):
    from datetime import time

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
    tool = next(e for e in events if e["kind"] == "tool_call")
    assert (tool["node"], tool["name"], tool["output"]["source"]) == ("weather", "weather_forecast", "fixture")
    steps = [(e["node"], e["name"]) for e in events if e["kind"] == "step"]
    assert ("plan", "build_itinerary") in steps and ("weather", "replan_for_rain") in steps
    assert ("weather", "select_candidates") in steps and ("weather", "validate") in steps
