from typing import Literal, Annotated, TypedDict
from dataclasses import field, dataclass

from langgraph.channels import UntrackedValue
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, trim_messages
from langgraph.graph.message import add_messages

from corga.agent.request import PlanRequest
from corga.domain.weather import WeatherReport
from corga.services.results import Result
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences

Path = Literal["plan", "ask", "attack", "out_of_scope"]
RECENT_MESSAGES = 5  # the two previous turns and this message


class Update(TypedDict, total=False):
    """What a node returns: only the state fields it changes.

    LangGraph merges them into the stored state.
    """

    messages: list[AnyMessage | RemoveMessage]
    path: Path
    plan_request: PlanRequest
    forecast: WeatherReport
    window: TimeWindow | None
    preferences: Preferences
    itinerary: Itinerary | None
    result: Result
    reply: str


@dataclass
class TurnState:
    """What a turn works on.

    `messages`, `window`, `preferences` and `itinerary` are checkpointed per session. The fields marked `UntrackedValue`
    live for one turn. Nodes receive it as an instance and return an `Update`.
    """

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)  # appended to, never overwritten
    path: Annotated[Path | None, UntrackedValue] = None  # set by `route`
    plan_request: Annotated[PlanRequest | None, UntrackedValue] = None  # set by `extract`
    forecast: Annotated[WeatherReport | None, UntrackedValue] = None  # set by the plan's forecast step
    window: TimeWindow | None = None  # sticky across turns
    preferences: Preferences = field(default_factory=Preferences)  # sticky across turns
    itinerary: Itinerary | None = None  # only ever a validated plan
    result: Annotated[Result | None, UntrackedValue] = None  # set by the path's last node, rendered by `respond`
    reply: Annotated[str, UntrackedValue] = ""  # the text the user reads

    @property
    def question(self) -> str:
        return next((m.text for m in reversed(self.messages) if isinstance(m, HumanMessage)), "")


def recent(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep the last five messages, starting on a user message."""
    return trim_messages(messages, strategy="last", token_counter=len, max_tokens=RECENT_MESSAGES, start_on="human")
