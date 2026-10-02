from typing import Literal
from datetime import date

from pydantic import Field, BaseModel, ConfigDict

from corga.domain.hours import OpeningHours

Setting = Literal["indoor", "outdoor", "mixed"]


class Location(BaseModel):
    latitude: float
    longitude: float


class POI(BaseModel):
    # by field name too: a checkpoint restores each place with POI(**model_dump())
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

    def is_open_on(self, day: date) -> bool:
        """Check the place opens that day. Unknown hours count as closed."""
        return self.opening_hours is not None and self.opening_hours.is_open_on(day)
