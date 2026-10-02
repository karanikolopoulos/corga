"""The session trace: a LangChain tracer writing one JSON line per turn, node, planning step, tool call, retrieval and
LLM call, and `traced`, which makes a function or method call a planning step."""

import inspect

from typing import Any, Self
from collections.abc import Callable

from langchain_core.runnables import RunnableLambda

from corga.tracing.view import STEP_TAG
from corga.tracing.tracer import JsonlTracer

__all__ = ["JsonlTracer", "traced"]


class traced[R]:
    """Run each call as a LangChain chain run named after the function and tagged as a step; works on methods too.

    The run takes its callbacks from the run it is called in (a graph node), so the session tracer records it with its
    arguments by name and its result.
    """

    def __init__(self, fn: Callable[..., R]):
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        step = RunnableLambda(self._call_with, name=self.fn.__name__).with_config(tags=[STEP_TAG])
        return step.invoke(self._named(args, kwargs))

    def __get__(self, instance: object, owner: type | None = None) -> Self:
        return type(self)(self.fn.__get__(instance, owner))

    def _named(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """The call's arguments by parameter name, as the trace shows them."""
        return inspect.signature(self.fn).bind(*args, **kwargs).arguments

    def _call_with(self, named: dict[str, Any]) -> R:
        return self.fn(**named)
