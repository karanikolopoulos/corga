from datetime import datetime
from functools import cached_property
from dataclasses import field, dataclass
from collections.abc import Callable

from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel

from corga.domain import timeutil
from corga.agent.state import ParsedRequest
from corga.agent.prompts import PromptStore
from corga.domain.weather import WeatherProvider
from corga.services.answers import Answers
from corga.services.itineraries import Itineraries


@dataclass
class Deps:
    """Everything the graph nodes need, built once by the entry point and passed in as plain objects."""

    llm: BaseChatModel  # a LangChain chat model; the tests script LangChain's fake chat model
    answers: Answers  # dataset facts: answers to questions and recommendations
    itineraries: Itineraries  # the visitor's plan: build, check, edit, adapt to the weather
    weather: WeatherProvider  # the forecast source the weather tool calls
    now: Callable[[], datetime] = timeutil.now  # the clock: today's date for the parser, the forecast range
    structured_method: str = "function_calling"  # `with_structured_output(method=...)`; see `parser`
    prompts: PromptStore = field(default_factory=PromptStore)  # the prompts, the state text and the reply layout

    @cached_property
    def parser(self) -> Runnable:
        """The LLM constrained to answer with a `ParsedRequest`, through `structured_method` (tool calling by default).

        Built on first use and kept for every later turn.
        """
        return self.llm.with_structured_output(ParsedRequest, method=self.structured_method)
