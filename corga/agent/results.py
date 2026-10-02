"""A turn's result: one model per kind, set by the handler. `respond` turns it into text: JSON for the LLM, the
reply template for the user."""

from typing import Self, Literal

from pydantic import BaseModel, ConfigDict, computed_field

from corga.data.hours import OpeningHours
from corga.data.models import POI, Setting
from corga.data.timeutil import Interval
from corga.weather.models import HourlyWeather
from corga.planning.models import Itinerary, Preferences, ValidationResult
from corga.planning.builder import FeasibilityReport
from corga.planning.candidates import matched_interests

RefusalReason = Literal["out_of_scope", "fabrication_or_ignore_dataset", "not_in_dataset"]


class Citation(BaseModel):
    """A dataset record the result cites, for the reply's Sources line."""

    id: str
    name: str

    @classmethod
    def of(cls, poi: POI) -> Self:
        return cls(id=poi.id, name=poi.name)


class Fact(BaseModel):
    """What the answer may say about a place; null opening hours stay null (unknown, not open).

    `score` is the search similarity, or None for a place the user named that the search did not return.
    """

    id: str
    score: float | None
    name: str
    category: str
    setting: Setting
    description: str
    opening_hours: OpeningHours | None
    source_urls: list[str]

    @classmethod
    def of(cls, poi: POI, score: float | None) -> Self:
        return cls(
            id=poi.id,
            score=round(score, 3) if score is not None else None,
            name=poi.name,
            category=poi.category,
            setting=poi.setting,
            description=poi.description,
            opening_hours=poi.opening_hours,
            source_urls=poi.source_urls,
        )


class Item(BaseModel):
    """A recommended place and which of the interests it matches."""

    id: str
    name: str
    category: str
    setting: Setting
    matched: list[str]
    family_suitable: bool | None

    @classmethod
    def of(cls, poi: POI, prefs: Preferences) -> Self:
        return cls(
            id=poi.id,
            name=poi.name,
            category=poi.category,
            setting=poi.setting,
            matched=matched_interests(poi, prefs.interests),
            family_suitable=poi.family_suitable,
        )


class Adaptation(BaseModel):
    """What a replan changed; the result carries the rebuilt itinerary itself."""

    model_config = ConfigDict(from_attributes=True)  # built from a `Replan`

    changed: bool
    replaced: list[str]
    note: str


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


class AnswerResult(BaseModel):
    kind: Literal["answer"] = "answer"
    question: str
    facts: list[Fact]
    unsupported_topics: list[str]
    unknown_places: list[str]

    @computed_field
    @property
    def citations(self) -> list[Citation]:
        return [Citation(id=fact.id, name=fact.name) for fact in self.facts]


class RecommendationResult(BaseModel):
    kind: Literal["recommendation"] = "recommendation"
    preferences: Preferences
    items: list[Item]

    @computed_field
    @property
    def citations(self) -> list[Citation]:
        return [Citation(id=item.id, name=item.name) for item in self.items]


class PlannedResult(BaseModel):
    """A result that may carry an itinerary; `planned` adds its validation."""

    itinerary: Itinerary | None = None
    validation: ValidationResult | None = None

    @computed_field
    @property
    def citations(self) -> list[Citation]:
        if self.itinerary is None:
            return []
        return [Citation.of(poi) for poi in self.itinerary.pois]


class PlanResult(PlannedResult):
    kind: Literal["plan"] = "plan"
    feasibility: FeasibilityReport | None = None  # the named places, checked
    adaptation: Adaptation | None = None  # an edit: the plan rebuilt for new preferences, or one stop replaced


class WeatherResult(PlannedResult):
    kind: Literal["weather"] = "weather"
    source: str
    range: Interval
    wet_periods: list[Interval]
    hours: list[HourlyWeather]
    adaptation: Adaptation | None = None  # the plan rebuilt for rain

    @computed_field
    @property
    def rain_expected(self) -> bool:
        return bool(self.wet_periods)


Result = (
    NeedInfoResult
    | RefusalResult
    | WeatherUnavailableResult
    | AnswerResult
    | RecommendationResult
    | PlanResult
    | WeatherResult
)
