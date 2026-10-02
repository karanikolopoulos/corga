"""Structured-metadata selection. Vector ranking is optional and only orders the filtered pool."""

from datetime import date
from collections.abc import Callable, Sequence

from corga.data.models import POI, Setting
from corga.planning.models import Preferences, PreferredSetting

Ranker = Callable[[str, set[str]], list[str]]

# Place settings each preference accepts: "outdoor" includes mixed places, "indoor" only indoor ones.
ACCEPTED_SETTINGS: dict[PreferredSetting, set[Setting]] = {"indoor": {"indoor"}, "outdoor": {"outdoor", "mixed"}}


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
    return prefs.setting is None or poi.setting in ACCEPTED_SETTINGS[prefs.setting]


def matched_interests(poi: POI, interests: Sequence[str]) -> list[str]:
    return [i for i in interests if i in INTEREST_RULES and INTEREST_RULES[i](poi)]


def select_candidates(pois: Sequence[POI], prefs: Preferences, ranker: Ranker | None = None) -> list[POI]:
    interests = prefs.interests
    known = [i for i in interests if i in INTEREST_RULES]
    pool = [p for p in pois if passes_filters(p, prefs) and (not known or matched_interests(p, known))]
    pool.sort(key=lambda p: p.id)
    if ranker and interests:
        order = {pid: rank for rank, pid in enumerate(ranker(" ".join(interests), {p.id for p in pool}))}
        pool.sort(key=lambda p: (order.get(p.id, len(order)), p.id))
    return pool


def recommendations(
    pois: Sequence[POI], prefs: Preferences, day: date | None = None, ranker: Ranker | None = None, limit: int = 5
) -> list[POI]:
    """The best `limit` candidates for the preferences, keeping only places open on `day` when one is given."""
    pool = select_candidates(pois, prefs, ranker)
    if day is not None:
        pool = [poi for poi in pool if poi.is_open_on(day)]
    return pool[:limit]
