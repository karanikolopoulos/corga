from typing import Self, Literal
from datetime import datetime

from pydantic import BaseModel, computed_field, field_validator, model_validator

from corga.data.models import Setting
from corga.data.timeutil import Interval

Severity = Literal["error", "warning"]
PreferredSetting = Literal["indoor", "outdoor"]  # what a visitor asks for; places are also labelled "mixed"


def normalize_interests(interests: list[str]) -> list[str]:
    """Interest keywords lowercased and trimmed, empty ones dropped."""
    return [cleaned for interest in interests if (cleaned := interest.strip().lower())]


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
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


class Stop(BaseModel):
    poi_id: str
    name: str
    setting: Setting
    walk_minutes: int
    arrive: datetime
    start: datetime
    end: datetime

    @property
    def span(self) -> Interval:
        """The visit itself, from start to end."""
        return Interval(self.start, self.end)


class Itinerary(BaseModel):
    window: TimeWindow
    stops: list[Stop] = []
    assumptions: list[str] = []

    @property
    def poi_ids(self) -> list[str]:
        return [s.poi_id for s in self.stops]


class Violation(BaseModel):
    code: str
    severity: Severity = "error"
    stop: int | None = None
    poi_id: str | None = None
    message: str


class ValidationResult(BaseModel):
    violations: list[Violation] = []

    @computed_field
    @property
    def ok(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)


class Preferences(BaseModel):
    interests: list[str] = []
    with_child: bool = False
    avoid_museums: bool = False
    setting: PreferredSetting | None = None  # None: any setting

    @field_validator("interests")
    @classmethod
    def _normalized(cls, interests: list[str]) -> list[str]:
        return normalize_interests(interests)

    def without_interests(self) -> Self:
        """The same constraints (child, museums, setting) with no topic preference."""
        return self.model_copy(update={"interests": []})

    def indoor(self) -> Self:
        """The same preferences restricted to indoor places, e.g. for rain."""
        return self.model_copy(update={"setting": "indoor"})
