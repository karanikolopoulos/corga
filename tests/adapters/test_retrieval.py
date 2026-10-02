import pytest

from langchain_core.embeddings import DeterministicFakeEmbedding

from corga.adapters.retrieval.index import COLLECTION, build_index
from corga.adapters.retrieval.search import Retriever
from corga.adapters.retrieval.documents import poi_to_document


@pytest.fixture
def retriever(store):
    return Retriever.from_pois(store.all(), DeterministicFakeEmbedding(size=32))


def test_document_carries_id_and_tags(store):
    doc = poi_to_document(store.get("CF-001"))
    assert doc.metadata == {"id": "CF-001", "name": "Old Fortress"}
    assert "venetian" in doc.page_content and "Old Fortress" in doc.page_content


def test_index_holds_every_poi_in_memory(store):
    index = build_index(store.all(), DeterministicFakeEmbedding(size=32))
    assert index.client.count(COLLECTION).count == 50


def test_the_retriever_indexes_the_pois_it_is_given(store, retriever):
    assert (retriever.size, retriever.k) == (50, 4)
    assert retriever.store.client.count(COLLECTION).count == 50


def test_search_returns_k_documents_with_clean_metadata(retriever):
    docs = retriever.search("fortress")
    assert len(docs) == 4 and len({d.metadata["id"] for d in docs}) == 4
    assert all(set(d.metadata) == {"id", "name", "score"} for d in docs)  # no Qdrant internals such as `_id`
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)  # cosine similarity, best first


def test_search_respects_allowed_ids(retriever):
    docs = retriever.search("fortress", allowed_ids={"CF-001", "CF-002"})
    assert {d.metadata["id"] for d in docs} == {"CF-001", "CF-002"}
    assert set(retriever.rank("fortress", {"CF-007"})) == {"CF-007"}
    assert retriever.rank("fortress", set()) == []


def test_closest_maps_place_ids_to_scores_best_first(retriever):
    scores = retriever.closest("fortress")
    assert len(scores) == 4 and list(scores.values()) == sorted(scores.values(), reverse=True)


@pytest.mark.model
def test_minilm_finds_venetian_history(store):
    from langchain_huggingface import HuggingFaceEmbeddings

    pois = store.all()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever.from_pois(pois, embeddings)
    venetian = {p.id for p in pois if "venetian" in p.tags}
    assert venetian & {d.metadata["id"] for d in retriever.search("Venetian history and fortifications")}
