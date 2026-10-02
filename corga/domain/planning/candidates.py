import logging

from datetime import date
from collections.abc import Callable, Sequence

from corga.domain.places import POI, Setting
from corga.domain.planning.models import Preferences, PreferredSetting

log = logging.getLogger(__name__)

Ranker = Callable[[str, set[str]], list[str]]

# the settings each preference accepts: "outdoor" includes mixed places.
ACCEPTED_SETTINGS: dict[PreferredSetting, set[Setting]] = {"indoor": {"indoor"}, "outdoor": {"outdoor", "mixed"}}


def _has_tag(*tags: str) -> Callable[[POI], bool]:
    return lambda poi: bool(set(tags) & set(poi.tags))


def _category(name: str) -> Callable[[POI], bool]:
    return lambda poi: poi.category == name


INTEREST_RULES: dict[str, Callable[[POI], bool]] = {
    "history": _category("history"),
    "venetian": _has_tag("venetian"),
    "museums": _has_tag("museum"),
    "art": _has_tag("art", "gallery", "painting"),
    "nature": _category("nature"),
    "food": _category("food"),
    "architecture": _has_tag("architecture"),
}


def passes_filters(poi: POI, prefs: Preferences) -> bool:
    """Check the place suits the child, museum and setting preferences.

    Args:
        poi: The place.
        prefs: The preferences.

    Returns:
        Whether it passes.
    """
    if prefs.with_child and poi.family_suitable is not True:
        return False
    if prefs.avoid_museums and "museum" in poi.tags:
        return False

    return prefs.setting is None or poi.setting in ACCEPTED_SETTINGS[prefs.setting]


def fits_interests(poi: POI, interests: Sequence[str]) -> bool:
    """Check the place matches one of the interests.

    Args:
        poi: The place.
        interests: The interests.

    Returns:
        Whether it matches. Any place does when there is no interest.
    """
    return not interests or any(INTEREST_RULES[interest](poi) for interest in interests)


def select_candidates(pois: Sequence[POI], prefs: Preferences, ranker: Ranker | None = None) -> list[POI]:
    """Select the places that pass the filters and match a known interest.

    Args:
        pois: The places.
        prefs: The preferences.
        ranker: Orders the places for the interests. None keeps id order.

    Returns:
        The candidates, ranked for the interests when there is a ranker, else by id.
    """
    interests = prefs.interests
    pool = [p for p in pois if passes_filters(p, prefs) and fits_interests(p, interests)]

    order: dict[str, int] = {}

    if ranker and interests:
        ranked = ranker(" ".join(interests), {p.id for p in pool})
        order = {pid: rank for rank, pid in enumerate(ranked)}

    pool.sort(key=lambda p: (order.get(p.id, len(order)), p.id))  # ranked places first, ties and the rest by id

    log.debug("candidates: %s", [poi.id for poi in pool])

    return pool


def recommendations(
    pois: Sequence[POI], prefs: Preferences, day: date | None = None, ranker: Ranker | None = None, limit: int = 5
) -> list[POI]:
    """Recommend the best candidates for the preferences.

    Args:
        pois: The places.
        prefs: The preferences.
        day: Only places open that day, when given.
        ranker: Orders the places for the interests.
        limit: How many to return.

    Returns:
        The best `limit` candidates.
    """
    pool = select_candidates(pois, prefs, ranker)

    if day is not None:
        pool = [poi for poi in pool if poi.is_open_on(day)]

    return pool[:limit]
