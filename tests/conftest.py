import json

from typing import Any
from datetime import date
from collections.abc import Iterable

import pytest

from langchain_core.messages import ToolCall, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from corga import DATASET_PATH
from corga.agent.state import ParsedRequest
from corga.domain.timeutil import at
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.models import Itinerary, TimeWindow
from corga.domain.planning.schedule import Schedule

MON, TUE, WED = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)  # the test days; the rain fixture covers WED
RULES = PlanningRules()
FRAMING = "Here is what I found."  # the framing text of every scripted reply


def window(day: date, start: str, end: str) -> TimeWindow:
    """A time window on `day` from `start` to `end` (HH:MM, Europe/Athens)."""
    return TimeWindow(start=at(day, start), end=at(day, end))


def itinerary(store, ids: list[str], scenario: TimeWindow) -> Itinerary:
    """The places `ids` visited in that order from the start of `scenario`, with the rules' assumptions."""
    stops = list(Schedule.of([store.get(i) for i in ids], scenario.start, RULES.walking))
    return Itinerary(window=scenario, stops=stops, assumptions=RULES.assumptions())


def read(path):
    """A trace file's lines, as dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def of_kind(events, kind):
    return [e for e in events if e["kind"] == kind]


@pytest.fixture(scope="session")
def store():
    from corga.adapters.poi_store import POIStore

    return POIStore.from_json(DATASET_PATH)


class ScriptedChatModel(GenericFakeChatModel):
    """LangChain's fake chat model with `bind_tools`, so `with_structured_output` parses its scripted tool calls."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(requests: Iterable[ParsedRequest]) -> ScriptedChatModel:
    """Each turn's parse gets the next request as a tool call; each reply's framing gets FRAMING."""
    messages = []
    for n, request in enumerate(requests, start=1):
        args = request.model_dump(mode="json", exclude_none=True)
        messages.append(AIMessage(content="", tool_calls=[ToolCall(name="ParsedRequest", args=args, id=f"call_{n}")]))
        messages.append(AIMessage(content=FRAMING))
    return ScriptedChatModel(messages=iter(messages))


class StubRetriever(BaseRetriever):
    """Search returns the configured ids; rank returns nothing, so candidates stay in id order."""

    store: Any
    ids: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.search(query)

    def search(self, query, k=None, allowed_ids=None):
        ids = [i for i in self.ids if allowed_ids is None or i in allowed_ids][: k or 4]
        return [
            Document(page_content="", metadata={"id": i, "name": self.store.get(i).name, "score": 1.0}) for i in ids
        ]

    def closest(self, query):
        return {doc.metadata["id"]: doc.metadata["score"] for doc in self.invoke(query)}

    def rank(self, query, allowed_ids):
        return []


@pytest.fixture
def make_deps(store):
    from corga.domain import timeutil
    from corga.agent.runtime import Deps
    from corga.services.steps import PlanningSteps
    from corga.services.places import PlaceFinder
    from corga.services.answers import Answers
    from corga.services.itineraries import Itineraries
    from corga.adapters.weather.fixture import FixtureWeather

    def factory(requests, hit_ids=("CF-001",), retriever=None, now=timeutil.now):
        retriever = retriever or StubRetriever(store=store, ids=list(hit_ids))
        places, steps = PlaceFinder(store, retriever), PlanningSteps(store, RULES, retriever.rank)
        return Deps(
            llm=scripted(requests),
            answers=Answers(places, steps),
            itineraries=Itineraries(steps, places),
            weather=FixtureWeather(),
            now=now,
        )

    return factory
