"""The templates in this folder: the two prompts sent to the LLM, the session state as text for the parse prompt, and
the markdown layout of every reply.

Prompts are plain text with LangChain `{name}` placeholders; the reply is a Jinja2 template, filled here with the
turn's result. Prompts are chosen by file name.
"""

from jinja2 import Environment, StrictUndefined, FileSystemLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from corga import TEMPLATES_DIR
from corga.agent.state import AgentState
from corga.agent.results import Result

# --- prompts ---


def parse_prompt(file: str = "parse.md") -> ChatPromptTemplate:
    """Inputs: `today`, `context` (`state_summary`) and `messages` (the recent conversation)."""
    system = (TEMPLATES_DIR / file).read_text()
    return ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder("messages")])


def respond_prompt(file: str = "respond.md") -> ChatPromptTemplate:
    """Input: `result`, the turn's result as JSON."""
    system = (TEMPLATES_DIR / file).read_text()
    return ChatPromptTemplate.from_messages([("system", system), ("human", "RESULT:\n{result}")])


def state_summary(state: AgentState) -> str:
    """The session state as short text for the parse prompt: time window, preferences, current stops."""
    window, itinerary = state.window, state.itinerary
    lines = [f"time window: {window.start:%A %d %B %Y %H:%M}-{window.end:%H:%M}" if window else "time window: none"]
    lines.append(f"preferences: {state.preferences.model_dump(exclude_defaults=True) or 'none'}")
    if itinerary and itinerary.stops:
        stops = "; ".join(f"{n}. {s.name}" for n, s in enumerate(itinerary.stops, 1))
        lines.append(f"current itinerary: {stops}")
    else:
        lines.append("current itinerary: none")
    return "\n".join(lines)


# --- reply ---

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,  # a line holding only {% ... %} leaves no empty line behind
    lstrip_blocks=True,
    undefined=StrictUndefined,  # a missing value raises an error
    autoescape=False,  # markdown, not HTML
)
_env.filters["hhmm"] = lambda t: f"{t:%H:%M}"
_env.filters["day"] = lambda t: f"{t:%A %d %B %Y}"
_env.filters["weekday_hour"] = lambda t: f"{t:%a %H:%M}"


def render_result(result: Result) -> str:
    """The factual part of the reply: times, walking legs, checks, forecasts, citations; empty sections left out."""
    return _env.get_template("reply.md.j2").render(result=result).strip()
