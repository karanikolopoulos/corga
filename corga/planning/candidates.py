"""Structured-metadata selection. Vector ranking is optional and only orders the filtered pool."""

from collections.abc import Callable, Sequence

from corga.data.models import POI
from corga.planning.models import Preferences

Ranker = Callable[[str, set[str]], list[str]]


def _has_tag(*tags: str) -> Callable[[POI], bool]:
    return lambda poi: bool(set(tags) & set(poi.tags))


def _category(name: str) -> Callable[[POI], bool]:
    return lambda poi: poi.category == name


INTEREST_RULES: dict[str, Callable[[POI], bool]] = {
    "history": _category("history"),
    "historical": _category("history"),
    "venetian": _has_tag("venetian"),
    "museum": _has_tag("museum"),
    "museums": _has_tag("museum"),
    "art": _has_tag("art", "gallery", "painting"),
    "nature": _category("nature"),
    "food": _category("food"),
    "architecture": _has_tag("architecture"),
}


def passes_filters(poi: POI, prefs: Preferences) -> bool:
    if prefs.with_child and poi.family_suitable is not True:
        return False
    if prefs.avoid_museums and "museum" in poi.tags:
        return False
    if prefs.outdoor_only and poi.setting != "outdoor":
        return False
    return not (prefs.indoor_only and poi.setting != "indoor")


def matched_interests(poi: POI, interests: Sequence[str]) -> list[str]:
    return [i for i in interests if i in INTEREST_RULES and INTEREST_RULES[i](poi)]


def select_candidates(pois: Sequence[POI], prefs: Preferences, ranker: Ranker | None = None) -> list[POI]:
    interests = [i.strip().lower() for i in prefs.interests]
    known = [i for i in interests if i in INTEREST_RULES]
    pool = [p for p in pois if passes_filters(p, prefs) and (not known or matched_interests(p, known))]
    pool.sort(key=lambda p: p.id)
    if ranker and interests:
        order = {pid: rank for rank, pid in enumerate(ranker(" ".join(interests), {p.id for p in pool}))}
        pool.sort(key=lambda p: (order.get(p.id, len(order)), p.id))
    return pool
