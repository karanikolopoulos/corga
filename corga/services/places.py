"""Finding dataset places: by the names a user gives, or by closeness in meaning to a question."""

from typing import NamedTuple
from collections.abc import Iterable

from corga.domain.places import POI
from corga.services.ports import PlaceSearch, PlaceCatalog


class Hit(NamedTuple):
    """A place the search returned, with its cosine similarity to the query (higher is closer)."""

    poi: POI
    score: float


class Lookup(NamedTuple):
    """Place names the user gave, split into dataset places and names the dataset does not have."""

    places: list[POI]  # in the order named, each once
    unknown: list[str]


def unique(pois: Iterable[POI]) -> list[POI]:
    """The places in order, each once."""
    return list({poi.id: poi for poi in pois}.values())


class PlaceFinder:
    """Dataset places by name or by meaning, over the dataset and the similarity search."""

    def __init__(self, store: PlaceCatalog, retriever: PlaceSearch):
        self.store = store
        self.retriever = retriever

    def search(self, query: str) -> list[Hit]:
        """The dataset places closest to the query, best first, with their similarity scores."""
        scores = self.retriever.closest(query)
        return [Hit(self.store.get(poi_id), score) for poi_id, score in scores.items()]

    def lookup(self, names: list[str]) -> Lookup:
        """Match place names against the dataset."""
        found = {name: self.store.find(name) for name in names}
        places = unique(poi for poi in found.values() if poi is not None)

        return Lookup(places, [name for name, poi in found.items() if poi is None])
