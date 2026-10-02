"""What the graph carries: the LLM's reading of each message, and the conversation state kept per session."""

import datetime as dt  # module import: the `date` field below would shadow a bare `date` type

from typing import Any, Literal, Annotated, TypedDict

from pydantic import Field, BaseModel, field_validator
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from corga.planning.models import Itinerary, TimeWindow, Preferences, PreferredSetting, normalize_interests

Intent = Literal["question", "recommend", "plan", "edit_plan", "weather", "out_of_scope"]
Result = dict[str, Any]  # a turn's structured outcome as JSON-ready data; `kind` says which one
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
        """The preferences this message states: the non-null flags and setting, and the interests if any."""
        updates = {name: getattr(self, name) for name in PREFERENCE_FIELDS if getattr(self, name) is not None}
        if self.interests:
            updates["interests"] = self.interests
        return updates

    @field_validator("interests")
    @classmethod
    def _normalized(cls, interests: list[str]) -> list[str]:
        return normalize_interests(interests)

    @field_validator("start_time", "end_time")
    @classmethod
    def _local_time(cls, value: dt.time | None) -> dt.time | None:
        """Drop any UTC offset a model adds ("13:00+03:00"): times are Europe/Athens local, applied by `at()`."""
        return value.replace(tzinfo=None) if value else value


class AgentState(TypedDict, total=False):
    """What the checkpointer keeps per session; only `messages` accumulates, the rest is overwritten each turn.

    Nodes return an `AgentState` holding only the keys they change; LangGraph merges it into the stored state.
    Every key is optional (`total=False`) so such a partial state type-checks.
    """

    messages: Annotated[list[AnyMessage], add_messages]  # appended to, never overwritten
    request: ParsedRequest  # this turn's interpretation
    window: TimeWindow | None  # sticky across turns
    preferences: Preferences  # sticky across turns
    itinerary: Itinerary | None  # only ever a validated plan
    result: Result  # rendered by `respond`
