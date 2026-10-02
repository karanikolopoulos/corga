"""What the graph carries: the LLM's reading of each message, and the conversation state kept per session."""

import datetime as dt  # module import: the `date` field below would shadow a bare `date` type

from typing import Any, Literal, Annotated, TypedDict
from dataclasses import field, dataclass

from pydantic import Field, BaseModel, field_validator
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from corga.services.results import Result
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences, PreferredSetting, normalize_interests

Intent = Literal["question", "recommend", "plan", "edit_plan", "weather", "out_of_scope"]
PREFERENCE_FIELDS = ("with_child", "avoid_museums", "setting")


class ParsedRequest(BaseModel):
    """A tourist's message about Corfu, as a structured request. Fields not stated in the message stay null.

    The LLM fills it through structured output; the class and field descriptions are its instructions.
    """

    intent: Intent = Field(description="What the user wants in this message.")
    question: str | None = Field(None, description="The factual question, for intent=question.")
    date: dt.date | None = Field(None, description="Visit or forecast date, only if stated or clearly implied.")
    start_time: dt.time | None = Field(None, description="Start of the time window (HH:MM), if stated.")
    end_time: dt.time | None = Field(None, description="End of the time window (HH:MM), if stated.")
    horizon_hours: int | None = Field(None, description="Forecast horizon for 'next N hours' requests.")
    interests: list[str] = Field(default_factory=list, description="Topic keywords stated in THIS message.")
    requested_places: list[str] = Field(default_factory=list, description="Place names the user explicitly named.")
    stop_index: int | None = Field(None, ge=1, description="1-based stop number to change, for edit_plan.")
    with_child: bool | None = Field(None, description="True if travelling with a child.")
    avoid_museums: bool | None = Field(None, description="True if the user wants no museums.")
    setting: PreferredSetting | None = Field(None, description="indoor or outdoor, if the user asks for that kind.")
    wants_fabrication: bool = Field(False, description="True if asked to ignore the dataset or invent places/facts.")

    @property
    def preference_updates(self) -> dict[str, Any]:
        """The preferences this message states: the non-null flags and setting, and the interests if any is not blank.

        The interests are passed on as written; `Preferences` cleans them.
        """
        updates = {name: getattr(self, name) for name in PREFERENCE_FIELDS if getattr(self, name) is not None}
        if normalize_interests(self.interests):
            updates["interests"] = self.interests
        return updates

    @field_validator("start_time", "end_time")
    @classmethod
    def _local_time(cls, value: dt.time | None) -> dt.time | None:
        """Drop any UTC offset a model adds ("13:00+03:00"): times are Europe/Athens local, applied by `at()`."""
        return value.replace(tzinfo=None) if value else value


class Update(TypedDict, total=False):
    """What a node returns: only the state fields it changes; LangGraph merges them into the stored state."""

    messages: list[AnyMessage]
    request: ParsedRequest
    window: TimeWindow | None
    preferences: Preferences
    itinerary: Itinerary | None
    result: Result


@dataclass
class AgentState:
    """What the checkpointer keeps per session; only `messages` accumulates, the rest is overwritten each turn.

    Nodes receive it as an instance and return an `Update` holding only the fields they change; LangGraph merges it
    into the stored state.
    """

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)  # appended to, never overwritten
    request: ParsedRequest | None = None  # this turn's interpretation, set by `parse` before any handler runs
    window: TimeWindow | None = None  # sticky across turns
    preferences: Preferences = field(default_factory=Preferences)  # sticky across turns
    itinerary: Itinerary | None = None  # only ever a validated plan
    result: Result | None = None  # set by the handler, rendered by `respond`

    @property
    def day(self) -> dt.date | None:
        return self.window.day if self.window else None

    @property
    def parsed(self) -> ParsedRequest:
        """This turn's request, which `parse` sets before any handler runs."""
        if self.request is None:
            raise RuntimeError("no parsed request: `parse` runs before every handler")
        return self.request

    @property
    def question(self) -> str:
        """The question as parsed, else the user's latest message as written."""
        latest = next((m.text for m in reversed(self.messages) if isinstance(m, HumanMessage)), "")
        return (self.request.question if self.request else None) or latest
