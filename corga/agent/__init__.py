"""The conversational agent: a LangGraph graph (parse → handler → respond), its dependencies and a chat session per
conversation."""

from corga.agent.runtime import Deps, ChatSession, build_graph

__all__ = ["ChatSession", "Deps", "build_graph"]
