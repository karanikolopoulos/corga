import pytest

from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from corga.paths import DATASET_PATH, RAIN_FIXTURE_PATH


@pytest.fixture(scope="session")
def store():
    from corga.data.store import POIStore

    return POIStore.from_json(DATASET_PATH)


class StubLLM:
    """Scripted ParsedRequests for `parse`; fixed framing text for `respond`."""

    def __init__(self, requests):
        self.requests = list(requests)

    def with_structured_output(self, schema, **kwargs):
        return StubParser(self)

    def invoke(self, prompt, *args, **kwargs):
        return AIMessage(content="Here is what I found.")


class StubParser:
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt, *args, **kwargs):
        return self._llm.requests.pop(0)


class StubRetriever:
    """Search returns the configured ids; rank returns nothing, so candidates stay in id order."""

    def __init__(self, store, ids):
        self._store, self._ids = store, list(ids)

    def search(self, query, k=None, allowed_ids=None):
        ids = [i for i in self._ids if allowed_ids is None or i in allowed_ids][: k or 4]
        return [
            Document(page_content="", metadata={"id": i, "name": self._store.get(i).name, "score": 1.0}) for i in ids
        ]

    def rank(self, query, allowed_ids):
        return []


@pytest.fixture
def make_deps(store):
    from corga.agent.deps import Deps
    from corga.weather.fixture import FixtureWeather

    def factory(requests, weather=None, hit_ids=("CF-001",)):
        weather = weather or FixtureWeather(RAIN_FIXTURE_PATH)
        return Deps(llm=StubLLM(requests), store=store, retriever=StubRetriever(store, hit_ids), weather=weather)

    return factory
