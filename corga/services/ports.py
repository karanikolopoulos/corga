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
        """Find the places nearest the query.

        Args:
            query: What to look for.

        Returns:
            The place ids with their cosine similarity, best first.
        """
        ...

    def rank(self, query: str, allowed_ids: set[str]) -> list[str]:
        """Order the allowed place ids by similarity to the query.

        Args:
            query: What to look for.
            allowed_ids: The places to order.

        Returns:
            The ids, best first.
        """
        ...
