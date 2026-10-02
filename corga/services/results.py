from typing import Self, Literal

from pydantic import BaseModel, ConfigDict, computed_field

from corga.domain.places import POI
from corga.domain.weather import WeatherReport
from corga.domain.planning.models import Itinerary, Preferences, ValidationResult
from corga.domain.planning.refill import Removal
from corga.domain.planning.builder import FeasibilityReport

RefusalReason = Literal["out_of_scope", "fabrication_or_ignore_dataset", "not_in_dataset"]


class Citation(BaseModel):
    id: str
    name: str

    @classmethod
    def of(cls, poi: POI) -> Self:
        return cls(id=poi.id, name=poi.name)


class Adaptation(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # built from a `Rebuild`

    changed: bool
    removed: list[Removal] = []
    added: list[str] = []  # place names


class NeedInfoResult(BaseModel):
    kind: Literal["need_info"] = "need_info"
    missing: str


class RefusalResult(BaseModel):
    kind: Literal["refusal"] = "refusal"
    reason: RefusalReason
    names: list[str] = []  # the unknown places, for not_in_dataset


class WeatherUnavailableResult(BaseModel):
    kind: Literal["weather_unavailable"] = "weather_unavailable"
    source: str
    reason: str


class PlanResult(BaseModel):
    """A plan: new, changed, or the check of named places."""

    kind: Literal["plan"] = "plan"
    itinerary: Itinerary | None = None
    validation: ValidationResult | None = None
    feasibility: FeasibilityReport | None = None  # the named places, checked
    adaptation: Adaptation | None = None  # an edit: the plan rebuilt for new preferences, or one stop replaced
    forecast: WeatherReport | None = None  # the forecast the plan was adapted to

    @computed_field
    @property
    def citations(self) -> list[Citation]:
        if self.itinerary is None:
            return []

        return [Citation.of(poi) for poi in self.itinerary.pois]


class PreferencesResult(BaseModel):
    """Preferences stored with no plan to change."""

    kind: Literal["preferences"] = "preferences"
    preferences: Preferences


class AnswerResult(BaseModel):
    """The ask agent's answer: its last message, with the cited ids inline."""

    kind: Literal["answer"] = "answer"
    answer: str


Result = NeedInfoResult | RefusalResult | PreferencesResult | AnswerResult | WeatherUnavailableResult | PlanResult
