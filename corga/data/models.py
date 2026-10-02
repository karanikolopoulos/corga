from typing import Literal

from pydantic import Field, BaseModel, ConfigDict

from corga.data.hours import OpeningHours

Setting = Literal["indoor", "outdoor", "mixed"]


class Location(BaseModel):
    latitude: float
    longitude: float


class POI(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    area: str
    category: str
    tags: list[str]
    location: Location
    description: str
    setting: Setting
    suggested_visit_minutes: int
    opening_hours: OpeningHours | None = Field(alias="opening_hours")
    family_suitable: bool | None = Field(alias="family_suitable")
    source_urls: list[str] = []
