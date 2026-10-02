from typing import Any
from datetime import datetime
from dataclasses import field, dataclass
from collections.abc import Callable

from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel

from corga.domain import timeutil
from corga.agent.prompts import PromptStore
from corga.services.places import PlaceFinder
from corga.services.weather import Weather
from corga.services.itineraries import Itineraries


@dataclass
class Deps:
    """Everything the nodes need, built once by the entry point."""

    llm: BaseChatModel  # a LangChain chat model, which the tests script with LangChain's fake chat model
    agent: CompiledStateGraph  # the ask agent over `llm`, built by the entry point with `build_agent`
    places: PlaceFinder  # the dataset's places: search, lookup by name, candidates and recommendations
    itineraries: Itineraries  # the visitor's plan: build, check, edit, adapt to the weather
    weather: Weather  # the forecast, for the weather tool and the plan's forecast step
    now: Callable[[], datetime] = timeutil.now  # the clock: today's date for the prompts, the forecast range
    structured_method: str | None = None  # None: the client's default method (the tests' scripted model)
    prompts: PromptStore = field(default_factory=PromptStore)  # the prompts, the state text and the reply layout
    _structured: dict[type[BaseModel], Runnable] = field(default_factory=dict, init=False, repr=False)

    def structured(self, schema: type[BaseModel]) -> Runnable:
        """Constrain the LLM to answer with `schema`, built once per schema.

        With `structured_method` None, the client picks the method.

        Args:
            schema: The model to answer with.

        Returns:
            A runnable returning the parsed model.
        """
        if schema not in self._structured:
            method: dict[str, Any] = {"method": self.structured_method} if self.structured_method else {}
            self._structured[schema] = self.llm.with_structured_output(schema, **method)

        return self._structured[schema]
