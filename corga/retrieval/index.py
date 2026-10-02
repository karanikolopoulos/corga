"""Qdrant index over the POI descriptions: in memory by default, or on a Qdrant server."""

from collections.abc import Sequence

from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings

from corga.data.models import POI
from corga.retrieval.documents import poi_to_document

COLLECTION = "pois"


def build_index(
    pois: Sequence[POI], embeddings: Embeddings, url: str | None = None, api_key: str | None = None
) -> QdrantVectorStore:
    """Embed the POIs into a fresh collection: in memory, or on the Qdrant server at `url` (`api_key`: Cloud)."""
    return QdrantVectorStore.from_documents(
        [poi_to_document(poi) for poi in pois],
        embeddings,
        location=":memory:" if url is None else None,
        url=url,
        api_key=api_key,
        collection_name=COLLECTION,
        force_recreate=True,  # a server keeps collections between runs; always start from the current dataset
    )
