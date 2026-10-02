import math

from dataclasses import dataclass

from corga.domain.places import Location

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: Location, b: Location) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


@dataclass(frozen=True)
class WalkingModel:
    """Walking time from the straight-line distance, a detour factor and a pace."""

    speed_kmh: float = 4.5
    detour_factor: float = 1.3

    def minutes(self, a: Location, b: Location) -> int:
        return math.ceil(haversine_km(a, b) * self.detour_factor / self.speed_kmh * 60)

    def __str__(self) -> str:
        return (
            f"Walking times are estimates: straight-line distance x {self.detour_factor} "
            f"at {self.speed_kmh} km/h, not a routed path."
        )
