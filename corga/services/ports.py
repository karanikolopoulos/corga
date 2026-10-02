"""What the service layer needs from outside it: the dataset and the similarity search. The adapters implement these
structurally, without importing them."""

from typing import Protocol

from corga.domain.places import POI


class PlaceCatalog(Protocol):
    """The curated dataset (`POIStore`)."""

    def get(self, poi_id: str) -> POI: ...

    def all(self) -> list[POI]: ...

    def find(self, name: str) -> POI | None: ...


class PlaceSearch(Protocol):
    """Similarity search over the place descriptions (`Retriever`)."""

    def closest(self, query: str) -> dict[str, float]:
        """The places nearest the query, best first: place id → cosine similarity."""
        ...

    def rank(self, query: str, allowed_ids: set[str]) -> list[str]:
        """The allowed place ids ordered by similarity to the query (a `Ranker` for candidate selection)."""
        ...
