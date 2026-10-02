from typing import Any
from dataclasses import field, dataclass

from pydantic import BaseModel

from corga.data.store import POIStore
from corga.planning.rules import PlanningRules
from corga.weather.models import WeatherProvider
from corga.retrieval.search import Retriever


@dataclass
class Deps:
    """Everything the graph nodes need, built once by the entry point and passed in as plain objects."""

    llm: Any  # a LangChain chat model (or a test double with the same two methods)
    store: POIStore
    retriever: Retriever
    weather: WeatherProvider
    rules: PlanningRules = field(default_factory=PlanningRules)
    structured_method: str | None = None  # `with_structured_output(method=...)`, for providers that need one

    def structured(self, schema: type[BaseModel]) -> Any:
        """The LLM constrained to answer with `schema`, using the provider's structured-output method if set."""
        kwargs = {"method": self.structured_method} if self.structured_method else {}
        return self.llm.with_structured_output(schema, **kwargs)
