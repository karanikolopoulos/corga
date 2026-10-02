from typing import Any, Self, Literal, Annotated
from datetime import date, datetime
from collections.abc import Iterable

from pydantic import BaseModel, PlainSerializer, computed_field, model_validator

from corga.domain.places import POI, Setting
from corga.domain.timeutil import Interval

Severity = Literal["error", "warning"]
ViolationCode = Literal[
    "EMPTY", "DUPLICATE", "SHORT_PLAN",  # the whole itinerary
    "OUTSIDE_WINDOW", "VISIT_TOO_SHORT", "UNKNOWN_HOURS", "CLOSED", "LONG_WAIT",  # a stop
    "TRAVEL_TOO_SHORT", "LEG_TOO_LONG",  # the walk into a stop
    "TIME_BUDGET", "CLOSED_DAY",  # a request for named places
]  # fmt: skip
PlaceRef = Annotated[POI, PlainSerializer(lambda poi: poi.id, return_type=str, when_used="json")]  # JSON: the id
PreferredSetting = Literal["indoor", "outdoor"]  # what a visitor asks for (places are also labelled "mixed")


class TimeWindow(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("time window must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("time window end must be after its start")

        return self

    @property
    def span(self) -> Interval:
        return Interval(self.start, self.end)

    @property
    def day(self) -> date:
        return self.start.date()

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


class Stop(BaseModel):
    """A visit to one place.

    In JSON the place is written as its id, followed by its name and setting.
    """

    poi: PlaceRef
    walk_minutes: int
    arrive: datetime
    start: datetime
    end: datetime

    @property
    def poi_id(self) -> str:
        return self.poi.id

    @computed_field
    @property
    def name(self) -> str:
        return self.poi.name

    @computed_field
    @property
    def setting(self) -> Setting:
        return self.poi.setting

    @property
    def span(self) -> Interval:
        return Interval(self.start, self.end)


class Itinerary(BaseModel):
    window: TimeWindow
    stops: list[Stop] = []
    assumptions: list[str] = []

    @property
    def pois(self) -> list[POI]:
        return [s.poi for s in self.stops]

    @property
    def poi_ids(self) -> list[str]:
        return [s.poi_id for s in self.stops]

    @property
    def minutes_used(self) -> int:
        """From the window's start to the last stop's end. 0 without stops."""
        if not self.stops:
            return 0

        return int((self.stops[-1].end - self.window.start).total_seconds() // 60)

    @property
    def half_filled(self) -> bool:
        return self.minutes_used * 2 >= self.window.minutes

    def has_stop(self, number: int) -> bool:
        return 1 <= number <= len(self.stops)

    def has_every_stop(self, numbers: Iterable[int]) -> bool:
        return all(self.has_stop(number) for number in numbers)


class Violation(BaseModel):
    code: ViolationCode
    severity: Severity = "error"
    stop: int | None = None
    poi_id: str | None = None
    message: str


class ValidationResult(BaseModel):
    violations: list[Violation] = []

    @computed_field
    @property
    def ok(self) -> bool:
        """No failed check is an error."""
        return not any(v.severity == "error" for v in self.violations)


class Preferences(BaseModel):
    interests: list[str] = []
    with_child: bool = False
    avoid_museums: bool = False
    setting: PreferredSetting | None = None  # None: any setting

    def merged(self, updates: dict[str, Any]) -> Self:
        """Apply `updates` to the preferences, validated as a new instance."""
        return self.model_validate(self.model_dump() | updates)

    def indoor(self) -> Self:
        return self.model_copy(update={"setting": "indoor"})
