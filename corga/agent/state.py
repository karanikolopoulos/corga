"""What the graph carries: the LLM's reading of each message, and the conversation state kept per session."""

import datetime as dt  # module import: the `date` field below would shadow a bare `date` type

from typing import Any, Literal, Annotated, TypedDict

from pydantic import Field, BaseModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from corga.planning.models import Itinerary, TimeWindow, Preferences

Intent = Literal["question", "recommend", "plan", "edit_plan", "weather", "out_of_scope"]
Result = dict[str, Any]  # a turn's structured outcome as JSON-ready data; `kind` says which one


class ParsedRequest(BaseModel):
    """A tourist's message about Corfu, as a structured request. Fields not stated in the message stay null.

    The only thing the LLM produces for control flow; the class and field descriptions are its instructions.
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
    outdoor_only: bool | None = Field(None, description="True if the user asks for outdoor places only.")
    wants_fabrication: bool = Field(False, description="True if asked to ignore the dataset or invent places/facts.")


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
