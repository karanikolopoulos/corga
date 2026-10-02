from uuid import uuid4
from typing import cast
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from corga.agent.deps import Deps
from corga.agent.turn import Turn
from corga.agent.graph import AgentGraph
from corga.agent.state import AgentState
from corga.agent.tracing import JsonlTracer


class ChatSession:
    """One conversation: one LangGraph thread and, unless tracing is off, one trace file."""

    def __init__(self, graph: AgentGraph, deps: Deps, tracer: JsonlTracer | None, session_id: str | None = None):
        self.id = session_id or uuid4().hex[:8]
        self.graph, self.deps, self.tracer = graph, deps, tracer
        # The thread id selects this conversation's checkpoint; the tracer sees every run of every call.
        self._config: RunnableConfig = {"configurable": {"thread_id": self.id}, "callbacks": [tracer] if tracer else []}
        self._event("session_started", session_id=self.id)

    @property
    def trace_path(self) -> Path | None:
        return self.tracer.path if self.tracer else None

    def send(self, text: str) -> str:
        """Run one turn and return the assistant's reply; Ctrl+C during the turn is recorded, then re-raised."""
        try:
            state = self.graph.invoke({"messages": [HumanMessage(text)]}, self._config, context=self.deps)
        except KeyboardInterrupt:
            self._event("turn_cancelled")
            raise
        return state["messages"][-1].text

    def state(self) -> AgentState:
        return cast(AgentState, self.graph.get_state(self._config).values)  # snapshots are typed as plain dicts

    def summary(self) -> str:
        """The session state as the parser sees it: time window, preferences, current itinerary."""
        return Turn(self.state(), self.deps).summary

    def close(self) -> None:
        if self.tracer:
            self.tracer.close()

    def _event(self, kind: str, **data: object) -> None:
        if self.tracer:
            self.tracer.event(kind, **data)
