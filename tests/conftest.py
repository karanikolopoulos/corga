import pytest

from corga.paths import DATASET_PATH


@pytest.fixture(scope="session")
def store():
    from corga.data.store import POIStore

    return POIStore.from_json(DATASET_PATH)
