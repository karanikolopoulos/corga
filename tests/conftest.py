import itertools

from typing import Any
from datetime import date
from collections.abc import Iterable

import pytest

from pydantic import BaseModel
from langchain_core.messages import ToolCall, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from corga import DATASET_PATH
from corga.agent.request import PlanRequest
from corga.domain.timeutil import at
from corga.agent.nodes.route import Route
from corga.domain.planning.rules import PlanningRules
from corga.domain.planning.models import Itinerary, TimeWindow
from corga.domain.planning.schedule import Schedule

MON, TUE, WED = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)  # the test days; the rain fixture covers WED
RULES = PlanningRules()
FRAMING = "Here is what I found."  # the framing text of every scripted reply


CALLS = itertools.count(1)


def window(day: date, start: str, end: str) -> TimeWindow:
    """Make a time window on `day` from `start` to `end` (HH:MM, Europe/Athens)."""
    return TimeWindow(start=at(day, start), end=at(day, end))


def itinerary(store, ids: list[str], scenario: TimeWindow) -> Itinerary:
    """Visit the places `ids` in order from the start of `scenario`."""
    stops = list(Schedule.of([store.get(i) for i in ids], scenario.start, RULES.walking))

    return Itinerary(window=scenario, stops=stops, assumptions=RULES.assumptions())


@pytest.fixture(scope="session")
def store():
    from corga.adapters.poi_store import POIStore

    return POIStore.from_json(DATASET_PATH)


class ScriptedChatModel(GenericFakeChatModel):
    """LangChain's fake chat model with `bind_tools`, for structured output.

    `with_structured_output` parses its scripted tool calls.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, **args: Any) -> AIMessage:
    """Script a tool call to `name` with `args`."""
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{next(CALLS)}")])


def answer(model: BaseModel) -> AIMessage:
    """Script the LLM's structured answer: a tool call named after the schema.

    It holds the fields that are not defaults.
    """
    args = model.model_dump(mode="json", exclude_defaults=True)

    return AIMessage(content="", tool_calls=[ToolCall(name=type(model).__name__, args=args, id=f"call_{next(CALLS)}")])


def routed(path: str) -> AIMessage:
    """Script the router's answer."""
    return answer(Route(path=path))


def plan_turn(request: PlanRequest) -> list[AIMessage]:
    """Script a plan turn: the route, the extraction and the framing of the reply."""
    return [routed("plan"), answer(request), AIMessage(content=FRAMING)]


def ask_turn(*agent: AIMessage) -> list[AIMessage]:
    """Script an ask turn: the route, then the agent's tool calls and answer."""
    return [routed("ask"), *agent]


def scripted(turns: Iterable[list[AIMessage]]) -> ScriptedChatModel:
    """Script the LLM's answers for a conversation, turn by turn."""
    return ScriptedChatModel(messages=iter([message for turn in turns for message in turn]))


class StubRetriever(BaseRetriever):
    """Search returns the configured ids. Rank returns nothing (id order)."""

    store: Any
    ids: list[str]

    def search(self, query, k=None, allowed_ids=None):
        ids = [i for i in self.ids if allowed_ids is None or i in allowed_ids][: k or 4]

        return [
            Document(page_content="", metadata={"id": i, "name": self.store.get(i).name, "score": 1.0}) for i in ids
        ]

    def closest(self, query):
        return {doc.metadata["id"]: doc.metadata["score"] for doc in self.invoke(query)}

    def rank(self, query, allowed_ids):
        return []

    def _get_relevant_documents(self, query, *, run_manager):
        return self.search(query)


@pytest.fixture
def make_deps(store):
    from corga.domain import timeutil
    from corga.agent.runtime import Deps
    from corga.agent.ask_agent import build_agent
    from corga.services.places import PlaceFinder
    from corga.services.weather import Weather
    from corga.services.itineraries import Itineraries
    from corga.adapters.weather.fixture import FixtureWeather

    def factory(turns=(), now=timeutil.now, llm=None):
        retriever = StubRetriever(store=store, ids=["CF-001"])
        places = PlaceFinder(store, retriever)
        llm = llm or scripted(turns)

        return Deps(
            llm=llm,
            agent=build_agent(llm),
            places=places,
            itineraries=Itineraries(places, RULES),
            weather=Weather(FixtureWeather()),
            now=now,
        )

    return factory
