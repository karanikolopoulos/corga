"""Named access to what a finished LangChain run carries."""

from typing import Any
from dataclasses import dataclass

from langchain_core.tracers.schemas import Run

STEP_TAG = "corga:step"  # the tag `traced` puts on a planning step's run


def _unwrapped(values: dict[str, Any] | None, key: str) -> Any:
    """A single plain value is stored by LangChain as {"input": value} / {"output": value}."""
    if values and set(values) == {key}:
        return values[key]

    return values


def _last_text(messages: list[Any] | None) -> str | None:
    if not messages:
        return None

    *_, last = messages

    return last.text


@dataclass(frozen=True)
class RunView:
    """A finished (or failed) run, read through named properties."""

    run: Run

    # --- what kind of run ---

    @property
    def is_turn(self) -> bool:
        """The graph call itself: one user turn."""
        return self.run.parent_run_id is None

    @property
    def is_node(self) -> bool:
        """A graph node: LangGraph names the node's run after the node."""
        return self.run.name == self.node

    @property
    def is_step(self) -> bool:
        """A planning step: a call through `traced`."""
        return STEP_TAG in (self.run.tags or [])

    # --- where it ran (LangGraph's metadata) ---

    @property
    def node(self) -> str | None:
        return self.run.metadata.get("langgraph_node")

    @property
    def step(self) -> int | None:
        return self.run.metadata.get("langgraph_step")

    @property
    def triggers(self) -> list[str] | None:
        return self.run.metadata.get("langgraph_triggers")

    # --- what happened ---

    @property
    def name(self) -> str:
        return self.run.name

    @property
    def input(self) -> Any:
        return _unwrapped(self.run.inputs, "input")

    @property
    def output(self) -> Any:
        return _unwrapped(self.run.outputs, "output")

    @property
    def latency_s(self) -> float | None:
        if (seconds := self.run.latency) is None:
            return None

        return round(seconds, 3)

    @property
    def error(self) -> str | None:
        """The error line of a failed run, without its traceback."""
        if not (error := self.run.error):
            return None

        message, *_ = error.split("Traceback", 1)

        return message.strip()

    # --- by run type ---

    @property
    def user_message(self) -> str | None:
        """A turn's input: the user's message."""
        return _last_text(self.run.inputs.get("messages"))

    @property
    def reply(self) -> str | None:
        """A turn's reply: the last message of the conversation it returns."""
        if not (outputs := self.run.outputs):
            return None

        return _last_text(outputs.get("messages"))

    @property
    def query(self) -> str | None:
        """A retrieval's query."""
        return self.run.inputs.get("query")

    @property
    def documents(self) -> list[dict[str, Any]]:
        """A retrieval's documents, as their metadata (id, name, score)."""
        if not (outputs := self.run.outputs):
            return []

        return [document.metadata for document in outputs.get("documents", [])]

    @property
    def model(self) -> str:
        """An LLM call's model name."""
        return self.run.metadata.get("ls_model_name") or self.run.name

    @property
    def usage(self) -> dict[str, Any] | None:
        """An LLM call's token usage, from the response message (kept serialised in LangChain's `lc` format)."""
        if not (outputs := self.run.outputs):
            return None

        [[generation, *_], *_] = outputs["generations"]
        message = generation.get("message", {})

        return message.get("kwargs", {}).get("usage_metadata")
