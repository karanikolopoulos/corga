"""The trace format: one record type per kind of line, each built from a `RunView`.

A failed run is the same record with `error` set. Records keep the objects a run carried (places, models, messages);
they become JSON when written (see `corga.agent.tracing.tracer`).
"""

from typing import Any, Self, ClassVar
from dataclasses import fields, dataclass

from corga.agent.tracing.view import RunView


@dataclass(frozen=True, kw_only=True)
class Record:
    """Fields every record has: how long the run took and, if it failed, its error line."""

    kind: ClassVar[str]
    latency_s: float | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        """The record's fields: kind first, then what ran, then how long it took and its error."""
        last = {"latency_s": self.latency_s, "error": self.error}
        what_ran = {field.name: getattr(self, field.name) for field in fields(self) if field.name not in last}

        return {"kind": self.kind} | what_ran | last

    @classmethod
    def of(cls, view: RunView) -> Self:
        """The record for a run: each field is read from the `RunView` property of the same name."""
        return cls(**{field.name: getattr(view, field.name) for field in fields(cls)})


@dataclass(frozen=True, kw_only=True)
class TurnRecord(Record):
    """One user turn: the message and the reply."""

    kind = "turn"
    input: str | None
    reply: str | None

    @classmethod
    def of(cls, view: RunView) -> Self:
        """A turn's input is the user's message, not the run's whole input."""
        return cls(input=view.user_message, reply=view.reply, latency_s=view.latency_s, error=view.error)


@dataclass(frozen=True, kw_only=True)
class GraphRecord(Record):
    """Fields of a run inside the graph: the node it belongs to and the graph step."""

    node: str | None
    step: int | None


@dataclass(frozen=True, kw_only=True)
class NodeRecord(GraphRecord):
    """A node invocation: what triggered it and its state update (its input, the whole state, is left out)."""

    kind = "node"
    triggers: list[str] | None
    output: Any


@dataclass(frozen=True, kw_only=True)
class CallRecord(GraphRecord):
    """A named call inside a node: its arguments and result."""

    name: str
    input: Any
    output: Any


@dataclass(frozen=True, kw_only=True)
class ToolRecord(CallRecord):
    """A tool call: its arguments and response."""

    kind = "tool"


@dataclass(frozen=True, kw_only=True)
class StepRecord(CallRecord):
    """A planning step (a `traced` call): its arguments by name and its result."""

    kind = "step"


@dataclass(frozen=True, kw_only=True)
class RetrievalRecord(GraphRecord):
    """A retrieval: the query and the found documents' metadata (id, name, score)."""

    kind = "retrieval"
    name: str
    query: str | None
    documents: list[dict[str, Any]]


@dataclass(frozen=True, kw_only=True)
class LLMRecord(GraphRecord):
    """An LLM call: the model and its token usage."""

    kind = "llm"
    model: str
    usage: dict[str, Any] | None
