import logging

from typing import Self
from collections.abc import Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from corga.domain.places import POI
from corga.adapters.retrieval.index import build_index

log = logging.getLogger(__name__)


class Retriever(BaseRetriever):
    """Similarity search over the indexed places.

    `invoke(query)` is the LangChain retriever call, recorded as a retrieval run. `search` and `rank` are the plain
    methods behind it. `rank` also orders the planner's candidates.
    """

    store: VectorStore
    size: int  # number of indexed POIs
    k: int = 4

    @classmethod
    def from_pois(
        cls,
        pois: Sequence[POI],
        embeddings: Embeddings,
        k: int = 4,
        *,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        qdrant_path: str | None = None,
    ) -> Self:
        """Index the places in memory, in a folder on disk or on a Qdrant server.

        Args:
            pois: The places to index.
            embeddings: The embedding model.
            k: How many places a search returns.
            qdrant_url: A Qdrant server.
            qdrant_api_key: The Qdrant Cloud key, for a server that needs one.
            qdrant_path: A folder that keeps the index between runs. With neither a URL nor a folder, the index is
                in memory.

        Returns:
            The retriever over the new index.
        """
        index = build_index(pois, embeddings, url=qdrant_url, api_key=qdrant_api_key, path=qdrant_path)

        return cls(store=index, size=len(pois), k=k)

    def search(self, query: str, k: int | None = None, allowed_ids: set[str] | None = None) -> list[Document]:
        """Search the places closest to the query.

        Args:
            query: What to look for.
            k: How many places to return. None uses the retriever's `k`.
            allowed_ids: Only these places, when given.

        Returns:
            The documents, best first, each with `id`, `name` and `score` (cosine similarity, higher is closer).
        """
        # score every POI, then keep the allowed ones
        pairs = self.store.similarity_search_with_score(query, k=self.size)
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={"id": doc.metadata["id"], "name": doc.metadata["name"], "score": score},
            )
            for doc, score in pairs
            if allowed_ids is None or doc.metadata["id"] in allowed_ids
        ]

        return docs[: k or self.k]

    def closest(self, query: str) -> dict[str, float]:
        """Find the nearest places through `invoke`, recorded as a retrieval run.

        Args:
            query: What to look for.

        Returns:
            The place ids with their similarity, best first.
        """
        hits = {doc.metadata["id"]: doc.metadata["score"] for doc in self.invoke(query)}

        log.debug("search %r: %s", query, hits)

        return hits

    def rank(self, query: str, allowed_ids: set[str]) -> list[str]:
        return [doc.metadata["id"] for doc in self.search(query, k=len(allowed_ids), allowed_ids=allowed_ids)]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.search(query)
