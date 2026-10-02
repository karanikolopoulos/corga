"""Everything in the templates folder: the two prompts sent to the LLM, and the markdown layout of every reply.

Prompts are plain text with LangChain `{name}` placeholders; the reply is a Jinja2 template, filled here with the
turn's result. Prompts are chosen by file name, so a variant is a new file rather than a code change.
"""

from jinja2 import Environment, StrictUndefined, FileSystemLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from corga.paths import TEMPLATES_DIR
from corga.data.store import POIStore
from corga.agent.state import Result
from corga.planning.models import Itinerary

# --- prompts ---


def parse_prompt(file: str = "parse.md") -> ChatPromptTemplate:
    """Inputs: `today`, `context` (`Turn.summary`) and `messages` (the recent conversation)."""
    system = (TEMPLATES_DIR / file).read_text()
    return ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder("messages")])


def respond_prompt(file: str = "respond.md") -> ChatPromptTemplate:
    """Input: `result`, the turn's result as JSON."""
    system = (TEMPLATES_DIR / file).read_text()
    return ChatPromptTemplate.from_messages([("system", system), ("human", "RESULT:\n{result}")])


# --- reply ---

# Result keys the reply template tests for; a result only carries the ones that apply to its kind.
OPTIONAL = {"feasibility": None, "validation": None, "items": [], "hours": [], "source": None}

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,  # a line holding only {% ... %} leaves no empty line behind
    lstrip_blocks=True,
    undefined=StrictUndefined,  # a typo in the template raises instead of rendering nothing
    autoescape=False,  # markdown, not HTML
)
_env.filters["hhmm"] = lambda t: f"{t:%H:%M}"
_env.filters["day"] = lambda t: f"{t:%A %d %B %Y}"


def render_result(result: Result, store: POIStore) -> str:
    """The factual part of the reply: times, walking legs, checks, forecasts, citations; empty sections left out."""
    itinerary = Itinerary.model_validate(result["itinerary"]) if result.get("itinerary") else None  # real datetimes
    sources = [f"{pid} {store.get(pid).name}" for pid in result.get("citations", [])]
    context = OPTIONAL | result | {"itinerary": itinerary, "sources": sources}
    return _env.get_template("reply.md.j2").render(context).strip()
