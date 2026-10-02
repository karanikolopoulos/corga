from langchain_core.documents import Document

from corga.data.models import POI


def poi_to_document(poi: POI) -> Document:
    """The text that gets embedded, with the POI id and name as metadata."""
    text = f"{poi.name}. Category: {poi.category}. Tags: {', '.join(poi.tags)}. Area: {poi.area}.\n{poi.description}"
    return Document(page_content=text, metadata={"id": poi.id, "name": poi.name})
