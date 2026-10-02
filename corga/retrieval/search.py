from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class Retriever:
    """Similarity search over the indexed POIs, optionally restricted to a set of POI ids."""

    def __init__(self, store: VectorStore, size: int, k: int = 4):
        self._store = store
        self._size = size  # number of indexed POIs
        self._k = k

    def search(self, query: str, k: int | None = None, allowed_ids: set[str] | None = None) -> list[Document]:
        """The `k` closest POIs, taken from `allowed_ids` when given.

        Each document keeps only `id` and `name` from the index, plus `score` (cosine similarity: higher is closer);
        store internals such as Qdrant's `_id` are dropped.
        """
        # Score every POI, then filter in Python: instant for 50 documents and independent of the store's
        # filter syntax (Qdrant could filter natively with a `FieldCondition` on `metadata.id`).
        pairs = self._store.similarity_search_with_score(query, k=self._size)
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={"id": doc.metadata["id"], "name": doc.metadata["name"], "score": score},
            )
            for doc, score in pairs
            if allowed_ids is None or doc.metadata["id"] in allowed_ids
        ]
        return docs[: k or self._k]

    def rank(self, query: str, allowed_ids: set[str]) -> list[str]:
        """Order the allowed POIs by similarity; satisfies `corga.planning.candidates.Ranker`."""
        return [doc.metadata["id"] for doc in self.search(query, k=len(allowed_ids), allowed_ids=allowed_ids)]
