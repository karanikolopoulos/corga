from uuid import uuid4
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from corga.agent.state import AgentState
from corga.agent.tracing import JsonlTracer
from corga.agent.templates import state_summary
from corga.agent.runtime.deps import Deps
from corga.agent.runtime.graph import AgentGraph


class ChatSession:
    """One conversation: one LangGraph thread and, unless tracing is off, one trace file."""

    def __init__(self, graph: AgentGraph, deps: Deps, tracer: JsonlTracer | None, session_id: str | None = None):
        self.id = session_id or uuid4().hex[:8]
        self.graph, self.deps, self.tracer = graph, deps, tracer
        # The thread id selects this conversation's checkpoint; the tracer receives every run of every call.
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
        """The conversation state after the last turn (snapshots hold plain dicts; the dataclass fills defaults)."""
        return AgentState(**self.graph.get_state(self._config).values)

    def summary(self) -> str:
        """The session state as the parser sees it: time window, preferences, current itinerary."""
        return state_summary(self.state())

    def close(self) -> None:
        """Drop this conversation's checkpoints from the checkpointer, then close the trace file."""
        self.graph.checkpointer.delete_thread(self.id)
        if self.tracer:
            self.tracer.close()

    def _event(self, kind: str, **data: object) -> None:
        if self.tracer:
            self.tracer.event(kind, **data)
