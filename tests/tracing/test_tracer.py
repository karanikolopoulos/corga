from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from corga.tracing import JsonlTracer, traced
from tests.conftest import WED, read, of_kind
from corga.domain.timeutil import at
from corga.domain.planning.models import TimeWindow


def test_llm_lines_record_model_and_token_usage(tmp_path):
    usage = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
    llm = GenericFakeChatModel(messages=iter([AIMessage("hi", usage_metadata=usage)]))
    tracer = JsonlTracer(tmp_path / "s.jsonl")
    llm.invoke("hello", config={"callbacks": [tracer]})
    tracer.close()
    [line] = of_kind(read(tmp_path / "s.jsonl"), "llm")
    assert line["model"] == "GenericFakeChatModel" and line["usage"] == usage and line["latency_s"] >= 0


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


def test_tracer_writes_places_as_their_ids(store, tmp_path):
    tracer = JsonlTracer(tmp_path / "s.jsonl")
    tracer.event("probe", places=[store.get("CF-001"), store.get("CF-007")])
    tracer.close()
    assert read(tmp_path / "s.jsonl")[0]["places"] == ["CF-001", "CF-007"]


def test_traced_records_functions_and_methods_as_runs_named_after_them(tmp_path):
    @traced
    def double(n: int) -> int:
        return 2 * n

    class Scaler:
        def __init__(self, factor: int):
            self.factor = factor

        @traced
        def scale(self, n: int) -> int:
            return double(n) * self.factor

    class State(TypedDict):
        out: int

    graph = StateGraph(State)
    graph.add_node("node", lambda _: {"out": Scaler(10).scale(n=3)})
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    tracer = JsonlTracer(tmp_path / "s.jsonl")
    assert graph.compile().invoke({"out": 0}, {"callbacks": [tracer]})["out"] == 60
    tracer.close()
    lines = {e["name"]: e for e in of_kind(read(tmp_path / "s.jsonl"), "step")}
    assert {line["node"] for line in lines.values()} == {"node"}  # both run inside the graph node
    assert (lines["double"]["input"], lines["double"]["output"]) == ({"n": 3}, 6)  # a function, called inside
    assert (lines["scale"]["input"], lines["scale"]["output"]) == ({"n": 3}, 60)  # a method: `self` is not recorded
