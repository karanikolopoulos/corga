from typing import NamedTuple
from datetime import date
from collections.abc import Iterable

from corga.domain.places import POI
from corga.services.ports import PlaceSearch, PlaceCatalog
from corga.domain.planning import candidates
from corga.domain.planning.models import Preferences


class Hit(NamedTuple):
    """A place the search returned, with its cosine similarity (higher is closer)."""

    poi: POI
    score: float


class Lookup(NamedTuple):
    """Place names the user gave: the dataset places, and the names it lacks."""

    places: list[POI]  # in the order named, each once
    unknown: list[str]


def unique(pois: Iterable[POI]) -> list[POI]:
    return list({poi.id: poi for poi in pois}.values())


class PlaceFinder:
    def __init__(self, store: PlaceCatalog, retriever: PlaceSearch):
        self.store = store
        self.retriever = retriever

    def search(self, query: str) -> list[Hit]:
        scores = self.retriever.closest(query)

        return [Hit(self.store.get(poi_id), score) for poi_id, score in scores.items()]

    def candidates(self, prefs: Preferences) -> list[POI]:
        return candidates.select_candidates(self.store.all(), prefs, self.retriever.rank)

    def recommend(self, prefs: Preferences, day: date | None) -> list[POI]:
        return candidates.recommendations(self.store.all(), prefs, day, self.retriever.rank)

    def lookup(self, names: list[str]) -> Lookup:
        """Match place names against the dataset.

        Args:
            names: The names as the user gave them.

        Returns:
            The dataset places, and the names it lacks.
        """
        found = {name: self.store.find(name) for name in names}
        places = unique(poi for poi in found.values() if poi is not None)

        return Lookup(places, [name for name, poi in found.items() if poi is None])
