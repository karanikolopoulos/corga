import json

from typing import Any
from pathlib import Path

from pydantic_core import to_jsonable_python
from langchain_core.tracers.schemas import Run
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from corga.domain.places import POI

FIELDS = {"name", "run_type", "tags", "start_time", "end_time", "inputs", "outputs", "error"}


def jsonable(obj: Any) -> Any:
    """Write a place as its id, anything else as JSON or its text."""
    if isinstance(obj, POI):
        return obj.id

    return to_jsonable_python(obj, fallback=str)


def tree(run: Run) -> dict[str, Any]:
    """Keep a run's readable fields and its child runs, which the run's own dump leaves out."""
    return run.model_dump(include=FIELDS) | {"child_runs": [tree(child) for child in run.child_runs]}


class JsonTracer(RunCollectorCallbackHandler):
    """Write a session's turns to one JSON file, each turn with its whole run tree, rewritten after every turn."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _persist_run(self, run: Run) -> None:
        super()._persist_run(run)

        turns = [tree(turn) for turn in self.traced_runs]
        self.path.write_text(json.dumps(turns, ensure_ascii=False, indent=1, default=jsonable), encoding="utf-8")
