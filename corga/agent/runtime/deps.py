from typing import Any
from functools import cached_property
from dataclasses import field, dataclass

from langchain_core.prompts import ChatPromptTemplate

from corga.agent.guide import Guide
from corga.agent.state import ParsedRequest
from corga.agent.templates import parse_prompt, respond_prompt


@dataclass
class Deps:
    """Everything the graph nodes need, built once by the entry point and passed in as plain objects."""

    llm: Any  # a LangChain chat model (or a test double with the same two methods)
    guide: Guide  # the planning steps, dataset lookups and forecast over the store, rules, retriever and weather
    structured_method: str = "function_calling"  # `with_structured_output(method=...)`; see `parser`
    parse_prompt: ChatPromptTemplate = field(default_factory=parse_prompt)  # the bundled prompts unless configured
    respond_prompt: ChatPromptTemplate = field(default_factory=respond_prompt)

    @cached_property
    def parser(self) -> Any:
        """The LLM constrained to answer with a `ParsedRequest`, through `structured_method` (tool calling by default).

        Built on first use and kept for every later turn.
        """
        return self.llm.with_structured_output(ParsedRequest, method=self.structured_method)
