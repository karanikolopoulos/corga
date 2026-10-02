"""A LangChain tracer that appends one JSON line per finished run of interest to the session's trace file.

`BaseTracer` sorts runs by type and calls one hook per type when a run finishes or fails; each hook writes one record.
Run types without a hook here (prompts, output parsers) are not written.
"""

import json

from typing import Any
from pathlib import Path

from pydantic_core import to_jsonable_python
from langchain_core.messages import BaseMessage
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from corga.data.models import POI
from corga.data.timeutil import now
from corga.agent.tracing.view import RunView
from corga.agent.tracing.records import (
    Record,
    LLMRecord,
    NodeRecord,
    StepRecord,
    ToolRecord,
    TurnRecord,
    RetrievalRecord,
)


def _default(obj: Any) -> Any:
    """JSON for what records carry besides plain data: messages, places (as ids), then pydantic's JSON for the rest
    (models, datetimes), with `str` for anything else."""
    if isinstance(obj, BaseMessage):
        return {"type": obj.type, "text": obj.text}
    if isinstance(obj, POI):
        return obj.id
    return to_jsonable_python(obj, fallback=str)


class JsonlTracer(BaseTracer):
    """One JSON line per finished turn, node, planning step, tool call, retrieval and LLM call, flushed at once."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: Record) -> None:
        self.event(**record.as_dict())

    def event(self, kind: str, **data: Any) -> None:
        """Append one line; also used directly for events outside a graph call (session start, cancelled turn)."""
        line = {"ts": now().isoformat(), "kind": kind, **data}
        self._file.write(json.dumps(line, ensure_ascii=False, default=_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self.event("session_closed")
            self._file.close()

    # --- BaseTracer hooks: each is called when a run of its type finishes; `*_error` when it fails ---

    def _persist_run(self, run: Run) -> None:
        """Required by `BaseTracer`; records are written as runs end, by the hooks below."""

    def _on_chain_end(self, run: Run) -> None:
        view = RunView(run)
        if view.is_turn:
            self.write(TurnRecord.of(view))
        elif view.is_step:
            self.write(StepRecord.of(view))
        elif view.is_node:
            self.write(NodeRecord.of(view))

    def _on_tool_end(self, run: Run) -> None:
        self.write(ToolRecord.of(RunView(run)))

    def _on_retriever_end(self, run: Run) -> None:
        self.write(RetrievalRecord.of(RunView(run)))

    def _on_llm_end(self, run: Run) -> None:
        self.write(LLMRecord.of(RunView(run)))

    _on_chain_error = _on_chain_end
    _on_tool_error = _on_tool_end
    _on_retriever_error = _on_retriever_end
    _on_llm_error = _on_llm_end
