"""Append-only JSONL trace per conversation, written by a LangChain tracer attached to every graph call.

The tracer receives each finished run (a node, a step or tool inside it, an LLM call, the whole turn) with its inputs,
outputs, timing and errors.
"""

import json

from typing import Any
from pathlib import Path
from datetime import date, time, datetime

from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from corga.agent.steps import STEP_TAG
from corga.data.timeutil import now


def _default(obj: Any) -> Any:
    """JSON for what runs carry besides plain data: messages, pydantic models, datetimes, sets."""
    if isinstance(obj, BaseMessage):
        return {"type": obj.type, "text": obj.text}
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _seconds(run: Run) -> float | None:
    return round((run.end_time - run.start_time).total_seconds(), 3) if run.end_time else None


def _error(run: Run) -> str | None:
    """The error line of a failed run, without its traceback."""
    return run.error.split("Traceback", 1)[0].strip() if run.error else None


def _unwrapped(values: dict[str, Any] | None, key: str) -> Any:
    """A run's inputs or outputs; a single plain value is stored by LangChain as {"input": ...} / {"output": ...}."""
    return values[key] if values and set(values) == {key} else values


def _usage(run: Run) -> dict[str, Any] | None:
    """Token usage of an LLM run; the run keeps the response message serialised (LangChain's `lc` format)."""
    message = run.outputs["generations"][0][0].get("message", {}) if run.outputs else {}
    return message.get("kwargs", {}).get("usage_metadata")


class JsonlTracer(BaseTracer):
    """Writes one JSON line per finished node, step, tool call, LLM call and turn of a session, flushed at once."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **data: Any) -> None:
        """Append one record; also used directly for events outside a graph call (session start, cancelled turn)."""
        record = {"ts": now().isoformat(), "kind": kind, **data}
        self._file.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self.event("session_closed")
            self._file.close()

    # --- runs, as LangChain reports them ---

    def _persist_run(self, run: Run) -> None:
        """Required by `BaseTracer`; runs are written as they end, by the `_on_*` hooks below."""

    def _on_chain_end(self, run: Run) -> None:
        node = run.metadata.get("langgraph_node")
        if run.parent_run_id is None:  # the graph call itself: one user turn
            reply = run.outputs["messages"][-1] if run.outputs and run.outputs.get("messages") else None
            self.event("turn", input=run.inputs.get("messages"), reply=reply, latency_s=_seconds(run))
        elif run.name == node:  # a graph node; prompts and the router also run as chains inside nodes
            step = run.metadata.get("langgraph_step")
            self.event("node", node=node, step=step, update=run.outputs, latency_s=_seconds(run))
        elif STEP_TAG in (run.tags or []):  # a named step of `corga.agent.steps`
            self.event(
                "step",
                node=node,
                name=run.name,
                input=_unwrapped(run.inputs, "input"),
                output=_unwrapped(run.outputs, "output"),
                latency_s=_seconds(run),
            )

    def _on_chain_error(self, run: Run) -> None:
        node = run.metadata.get("langgraph_node")
        if run.name == node:  # the failing node; its prompt, sequence and the graph report the same error again
            self.event("error", node=node, error=_error(run))
        elif STEP_TAG in (run.tags or []):
            self.event("error", node=node, step=run.name, error=_error(run))

    def _on_tool_end(self, run: Run) -> None:
        self.event(
            "tool_call",
            node=run.metadata.get("langgraph_node"),
            name=run.name,
            input=run.inputs,
            output=_unwrapped(run.outputs, "output"),
            latency_s=_seconds(run),
        )

    def _on_tool_error(self, run: Run) -> None:
        self.event(
            "tool_error", node=run.metadata.get("langgraph_node"), name=run.name, input=run.inputs, error=_error(run)
        )

    def _on_llm_end(self, run: Run) -> None:
        self.event(
            "llm_call",
            node=run.metadata.get("langgraph_node"),
            model=run.metadata.get("ls_model_name") or run.name,
            latency_s=_seconds(run),
            usage=_usage(run),
        )

    def _on_llm_error(self, run: Run) -> None:
        model = run.metadata.get("ls_model_name") or run.name
        self.event("llm_error", node=run.metadata.get("langgraph_node"), model=model, error=_error(run))
