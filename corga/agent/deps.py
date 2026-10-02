from typing import Any
from functools import cached_property
from dataclasses import field, dataclass

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from corga.data.store import POIStore
from corga.agent.steps import Steps
from corga.planning.rules import PlanningRules
from corga.weather.models import WeatherProvider
from corga.agent.templates import parse_prompt, respond_prompt
from corga.retrieval.search import Retriever


@dataclass
class Deps:
    """Everything the graph nodes need, built once by the entry point and passed in as plain objects."""

    llm: Any  # a LangChain chat model (or a test double with the same two methods)
    store: POIStore
    retriever: Retriever
    weather: WeatherProvider
    rules: PlanningRules = field(default_factory=PlanningRules)
    structured_method: str = "function_calling"  # `with_structured_output(method=...)`; see `structured`
    parse_prompt: ChatPromptTemplate = field(default_factory=parse_prompt)  # the bundled prompts unless configured
    respond_prompt: ChatPromptTemplate = field(default_factory=respond_prompt)

    @cached_property
    def steps(self) -> Steps:
        """The traced operations (weather tool, planning steps) over these dependencies."""
        return Steps(self)

    def structured(self, schema: type[BaseModel]) -> Any:
        """The LLM constrained to answer with `schema`, through `structured_method` (tool calling by default)."""
        return self.llm.with_structured_output(schema, method=self.structured_method)
