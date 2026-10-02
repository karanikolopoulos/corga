"""Shared setup for the service tests: the in-memory dataset store and a plain search fake, no LangChain."""

import pytest

from tests.conftest import RULES
from corga.services.steps import PlanningSteps
from corga.services.places import PlaceFinder


class FakeSearch:
    """A `PlaceSearch` with fixed scores; ranking keeps no order, so candidates stay in id order."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def closest(self, query: str) -> dict[str, float]:
        return self.scores

    def rank(self, query: str, allowed_ids: set[str]) -> list[str]:
        return []


@pytest.fixture
def search():
    return FakeSearch({"CF-001": 0.74, "CF-002": 0.61})


@pytest.fixture
def places(store, search):
    return PlaceFinder(store, search)


@pytest.fixture
def steps(store, search):
    return PlanningSteps(store, RULES, search.rank)
