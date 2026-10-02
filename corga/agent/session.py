from uuid import uuid4
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from corga.agent.deps import Deps
from corga.agent.graph import AgentGraph
from corga.agent.state import AgentState
from corga.agent.tracing import JsonlTracer


class ChatSession:
    """One conversation: one LangGraph thread, one trace file. The REPL and the eval runner both drive this."""

    def __init__(self, graph: AgentGraph, deps: Deps, tracer: JsonlTracer, session_id: str | None = None):
        self.id = session_id or uuid4().hex[:8]
        self.graph, self.deps, self.tracer = graph, deps, tracer
        # The thread id selects this conversation's checkpoint; the tracer sees every run of every call.
        self._config: RunnableConfig = {"configurable": {"thread_id": self.id}, "callbacks": [tracer]}
        tracer.event("session_started", session_id=self.id)

    def send(self, text: str) -> str:
        """Run one turn and return the assistant's reply."""
        state = self.graph.invoke({"messages": [HumanMessage(text)]}, self._config, context=self.deps)
        return state["messages"][-1].text

    def state(self) -> AgentState:
        return cast(AgentState, self.graph.get_state(self._config).values)  # snapshots are typed as plain dicts

    def close(self) -> None:
        self.tracer.close()
