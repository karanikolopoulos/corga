import pytest

from langchain_core.embeddings import DeterministicFakeEmbedding

from corga.adapters.retrieval.search import Retriever


@pytest.fixture
def retriever(store):
    return Retriever.from_pois(store.all(), DeterministicFakeEmbedding(size=32))


def test_search_respects_allowed_ids(retriever):
    docs = retriever.search("fortress", allowed_ids={"CF-001", "CF-002"})

    assert {d.metadata["id"] for d in docs} == {"CF-001", "CF-002"}
    assert set(retriever.rank("fortress", {"CF-007"})) == {"CF-007"}
    assert retriever.rank("fortress", set()) == []


class CountingEmbedding(DeterministicFakeEmbedding):
    """The fake embedder, counting the documents it embeds."""

    embedded: int = 0
    model_name: str = "fake"

    def embed_documents(self, texts):
        self.embedded += len(texts)

        return super().embed_documents(texts)


def test_a_folder_keeps_the_index_until_the_model_or_the_places_change(store, tmp_path):
    def index(embeddings: CountingEmbedding, pois) -> tuple[list[str], int]:
        """Index `pois` in the folder: its collections, and the texts embedded, a probe for the vector size included."""
        before = embeddings.embedded
        retriever = Retriever.from_pois(pois, embeddings, qdrant_path=str(tmp_path))
        names = [c.name for c in retriever.store.client.get_collections().collections]
        retriever.store.client.close()

        return names, embeddings.embedded - before

    minilm, other = CountingEmbedding(size=32), CountingEmbedding(size=32, model_name="other")
    built, _ = index(minilm, store.all())

    assert index(minilm, store.all()) == (built, 1)  # reopened: the probe only
    assert index(other, store.all())[1] == 51  # another model: rebuilt
    assert index(minilm, store.all()[:10])[1] == 11  # other places: rebuilt

    names, _ = index(minilm, store.all()[:10])

    assert len(names) == 1  # the stale collections are gone
