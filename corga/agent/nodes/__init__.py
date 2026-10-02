"""Graph nodes. Each takes the state plus the runtime (whose context is `Deps`) and returns an `Update`.

`result` is the turn's structured outcome: the handlers fill it, `respond` frames it (LLM) and renders it (template).
Handlers call the planning steps, dataset lookups and forecast through `runtime.context.guide`.
"""

from corga.agent.nodes.plan import plan, edit_plan
from corga.agent.nodes.parse import parse
from corga.agent.nodes.answer import answer, recommend
from corga.agent.nodes.respond import refuse, respond
from corga.agent.nodes.weather import weather

__all__ = ["answer", "edit_plan", "parse", "plan", "recommend", "refuse", "respond", "weather"]
