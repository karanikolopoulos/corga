"""Append-only JSONL trace per conversation, written by a LangChain tracer attached to every graph call.

A tracer receives each finished run (a node, an LLM call, the whole turn) with its inputs, outputs, timing and
errors; LangSmith's tracer works the same way, so both can be attached side by side.
"""

import json

from typing import Any
from pathlib import Path
from datetime import date, time, datetime

from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

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


def _usage(run: Run) -> dict[str, Any] | None:
    """Token usage of an LLM run; the run keeps the response message serialised (LangChain's `lc` format)."""
    message = run.outputs["generations"][0][0].get("message", {}) if run.outputs else {}
    return message.get("kwargs", {}).get("usage_metadata")


class JsonlTracer(BaseTracer):
    """Writes one JSON line per finished node, LLM call and turn of a session, flushed so a crash loses nothing."""

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
        """Required by `BaseTracer` (called once per finished top-level run); each run is written as it ends instead."""

    def _on_chain_end(self, run: Run) -> None:
        node = run.metadata.get("langgraph_node")
        if run.parent_run_id is None:  # the graph call itself: one user turn
            reply = run.outputs["messages"][-1] if run.outputs and run.outputs.get("messages") else None
            self.event("turn", input=run.inputs.get("messages"), reply=reply, latency_s=_seconds(run))
        elif run.name == node:  # a graph node; prompts and the router also run as chains inside nodes
            step = run.metadata.get("langgraph_step")
            self.event("node", node=node, step=step, update=run.outputs, latency_s=_seconds(run))

    def _on_chain_error(self, run: Run) -> None:
        self.event("error", node=run.metadata.get("langgraph_node"), run=run.name, error=run.error)

    def _on_llm_end(self, run: Run) -> None:
        self.event(
            "llm_call",
            node=run.metadata.get("langgraph_node"),
            model=run.metadata.get("ls_model_name") or run.name,
            latency_s=_seconds(run),
            usage=_usage(run),
        )

    def _on_llm_error(self, run: Run) -> None:
        self.event("error", node=run.metadata.get("langgraph_node"), run=run.name, error=run.error)
