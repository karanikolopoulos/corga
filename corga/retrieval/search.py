from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class Retriever(BaseRetriever):
    """Similarity search over the indexed POIs, optionally restricted to a set of POI ids.

    `invoke(query)` is the LangChain retriever call (recorded as a retrieval run); `search` and `rank` are the plain
    methods behind it, also used by candidate ranking.
    """

    store: VectorStore
    size: int  # number of indexed POIs
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.search(query)

    def search(self, query: str, k: int | None = None, allowed_ids: set[str] | None = None) -> list[Document]:
        """The `k` closest POIs, taken from `allowed_ids` when given.

        Each document keeps only `id` and `name` from the index, plus `score` (cosine similarity: higher is closer);
        store internals such as Qdrant's `_id` are dropped.
        """
        # Score every POI, then keep the allowed ones.
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

    def rank(self, query: str, allowed_ids: set[str]) -> list[str]:
        """Order the allowed POIs by similarity; satisfies `corga.planning.candidates.Ranker`."""
        return [doc.metadata["id"] for doc in self.search(query, k=len(allowed_ids), allowed_ids=allowed_ids)]
