import time
import logging

from uuid import uuid4
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from corga.agent.state import TurnState
from corga.agent.runtime.deps import Deps
from corga.agent.runtime.tracer import JsonTracer

log = logging.getLogger(__name__)


class ChatSession:
    """One conversation: one LangGraph thread and, when traced, one trace file."""

    def __init__(self, graph: CompiledStateGraph, deps: Deps, tracer: JsonTracer | None, session_id: str | None = None):
        self.id = session_id or uuid4().hex[:8]
        self.graph, self.deps, self.tracer = graph, deps, tracer
        self.last = TurnState()  # the state as the last turn ended, its own fields included

        # the thread id selects the checkpoints and the tracer receives every run
        self._config: RunnableConfig = {"configurable": {"thread_id": self.id}, "callbacks": [tracer] if tracer else []}

    @property
    def trace_path(self) -> Path | None:
        return self.tracer.path if self.tracer else None

    def send(self, text: str) -> str:
        started = time.monotonic()

        log.info("turn started (session %s)", self.id)
        log.debug("user: %s", text)

        out = self.graph.invoke({"messages": [HumanMessage(text)]}, self._config, context=self.deps)

        log.info("turn done in %.1f s (session %s)", time.monotonic() - started, self.id)

        self.last = TurnState(**out)

        return self.last.reply

    def state(self) -> TurnState:
        return TurnState(**self.graph.get_state(self._config).values)

    def summary(self) -> str:
        return self.deps.prompts["state"].render(state=self.state())

    def close(self) -> None:
        """Drop this conversation's checkpoints."""
        self.graph.checkpointer.delete_thread(self.id)
