import hashlib
import logging

from collections.abc import Sequence

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from corga.domain.places import POI
from corga.adapters.retrieval.documents import poi_to_document

log = logging.getLogger(__name__)

COLLECTION = "pois"


def fingerprint(docs: Sequence[Document], embeddings: Embeddings) -> str:
    """Name what the collection was built from: the embedding model and the documents."""
    model = getattr(embeddings, "model_name", type(embeddings).__name__)
    digest = hashlib.sha256(model.encode())

    for doc in docs:
        digest.update(doc.page_content.encode())

    return digest.hexdigest()[:12]


def build_index(
    pois: Sequence[POI],
    embeddings: Embeddings,
    url: str | None = None,
    api_key: str | None = None,
    path: str | None = None,
) -> QdrantVectorStore:
    """Embed the places into a collection, or reopen the one a folder already holds.

    Args:
        pois: The places to index.
        embeddings: The embedding model.
        url: A Qdrant server, used before `path`. With neither the index stays in memory.
        api_key: The Qdrant Cloud key, for a server that needs one.
        path: A folder for a local Qdrant on disk. Its collection is reused while the embedding model and the places
            are the same, and rebuilt when either changes.

    Returns:
        The vector store over the collection.
    """
    docs = [poi_to_document(poi) for poi in pois]

    if path and not url:
        return on_disk(docs, embeddings, path)

    return QdrantVectorStore.from_documents(
        docs,
        embeddings,
        location=":memory:" if not url else None,
        url=url or None,
        api_key=api_key,
        collection_name=COLLECTION,
        force_recreate=True,  # a server keeps collections between runs: always start from the current dataset
    )


def on_disk(docs: Sequence[Document], embeddings: Embeddings, path: str) -> QdrantVectorStore:
    """Reopen the folder's collection for this model and these places, or build it and drop the stale ones."""
    name = f"{COLLECTION}-{fingerprint(docs, embeddings)}"
    client = QdrantClient(path=path)

    if client.collection_exists(name):
        log.info("embeddings: reused %s in %s", name, path)

        return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)

    for stale in client.get_collections().collections:
        client.delete_collection(stale.name)

    client.close()  # from_documents opens the folder itself; a local Qdrant folder takes one client at a time

    log.info("embeddings: built %s in %s", name, path)

    return QdrantVectorStore.from_documents(docs, embeddings, path=path, collection_name=name)
