import datetime as dt  # module import: the `date` field below would shadow a bare `date` type

from typing import Any, Literal

from pydantic import Field, BaseModel, field_validator

from corga.domain.timeutil import at
from corga.domain.planning.models import TimeWindow, Preferences
from corga.domain.planning.refill import Change

Interest = Literal["history", "venetian", "museums", "art", "nature", "food", "architecture"]
Stated = Literal["yes", "no", "not_stated"]
Source = Literal["forecast"]  # the plan's data steps
SettingChoice = Literal["indoor", "outdoor", "any", "not_stated"]


def preference_updates(
    interests: list[str], child: Stated, avoid_museums: Stated, setting: SettingChoice
) -> dict[str, Any]:
    """Turn the stated preferences into `Preferences` fields.

    Args:
        interests: The interests stated. Empty when none.
        child: Whether a child comes along, or `not_stated`.
        avoid_museums: Whether to leave museums out, or `not_stated`.
        setting: `indoor`, `outdoor`, `any`, or `not_stated`.

    Returns:
        The fields to update. What is not stated is left out.
    """
    updates: dict[str, Any] = {}

    if child != "not_stated":
        updates["with_child"] = child == "yes"

    if avoid_museums != "not_stated":
        updates["avoid_museums"] = avoid_museums == "yes"

    if setting != "not_stated":
        updates["setting"] = None if setting == "any" else setting

    if interests:
        updates["interests"] = list(interests)

    return updates


class PlanRequest(BaseModel):
    """A message about the visit's plan, as the LLM reads it.

    Only what the message states.
    """

    date: dt.date | None = Field(
        default=None,
        description="The visit's day.",
    )
    start_time: dt.time | None = Field(
        default=None,
        description="When the visit starts.",
    )
    end_time: dt.time | None = Field(
        default=None,
        description="When the visit ends.",
    )
    interests: list[Interest] = Field(
        default=[],
        description="Topics the user wants the visit to cover.",
    )
    child: Stated = Field(
        default="not_stated",
        description="Whether a child comes along.",
    )
    avoid_museums: Stated = Field(
        default="not_stated",
        description="Whether to leave museums out.",
    )
    setting: SettingChoice = Field(
        default="not_stated", description="The kind of place the user wants: indoor, outdoor, or either (any)."
    )
    places: list[str] = Field(
        default=[],
        description="Specific places the user wants to visit.",
    )
    replace_stops: list[int] = Field(
        default=[], description="Numbers of the current plan's stops to replace, 1 being the first."
    )
    keep_other_stops: Stated = Field(
        default="not_stated", description="Whether the other stops of the current plan may change."
    )
    sources: list[Source] = Field(
        default=[],
        description="Data to fetch for the plan: the forecast.",
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def _local_time(cls, value: dt.time | None) -> dt.time | None:
        """Drop the UTC offset that JSON Schema's `time` format carries."""
        return value.replace(tzinfo=None) if value else value

    @property
    def preference_updates(self) -> dict[str, Any]:
        return preference_updates(self.interests, self.child, self.avoid_museums, self.setting)

    @property
    def states_a_time(self) -> bool:
        return any((self.date, self.start_time, self.end_time))

    @property
    def states_a_whole_window(self) -> bool:
        return all((self.date, self.start_time, self.end_time))

    @property
    def about_a_plan(self) -> bool:
        """Anything but preferences.

        A time, places, stops to replace, what to keep, or data to fetch.
        """
        about = (self.places, self.replace_stops, self.sources)

        return self.states_a_time or any(about) or self.keep_other_stops != "not_stated"

    @property
    def only_preferences(self) -> bool:
        return bool(self.preference_updates) and not self.about_a_plan


def resolve_window(prev: TimeWindow | None, req: PlanRequest) -> TimeWindow | None:
    """Replace the stated parts of the stored window.

    Args:
        prev: The stored window.
        req: The plan request.

    Returns:
        The new window. `prev` when nothing is stated or the edit is inverted. None for a first window that lacks the
        date, the start or the end.
    """
    if not req.states_a_time:
        return prev  # nothing stated, keep the window
    if prev is None and not req.states_a_whole_window:
        return None  # a first window needs all three parts

    day = req.date or prev.day
    start = at(day, req.start_time or prev.start.time())
    end = at(day, req.end_time or prev.end.time())

    return TimeWindow(start=start, end=end) if end > start else prev  # an inverted edit is ignored


def merge_preferences(prefs: Preferences, req: PlanRequest) -> Preferences:
    return prefs.merged(req.preference_updates)


def change_of(request: PlanRequest) -> Change:
    """Read what the request asks of the stored plan: stated preferences recheck the stops, unless it names stops."""
    return Change(
        replace=frozenset(request.replace_stops),
        keep_others=request.keep_other_stops != "no",
        recheck_preferences=bool(request.preference_updates) and not request.replace_stops,
    )
